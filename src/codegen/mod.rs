//! Code generation using Cranelift
//!
//! Compiles HIR to machine code via Cranelift JIT.

use crate::error::{BoltError, Result};
use crate::hir::{
    self, BinaryOp, Block as HirBlock, Crate, DefId, Expr, ExprKind, FloatType, Function,
    IntType, ItemKind, Literal, Pattern, PatternKind, Stmt, StmtKind, Type as HirType,
    TypeKind, UintType, UnaryOp,
};
use cranelift::prelude::{
    codegen, settings, types, AbiParam, Configurable, FunctionBuilder, FunctionBuilderContext,
    InstBuilder, IntCC, Value,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use indexmap::IndexMap;
use std::collections::HashMap;

pub struct CodeGenerator {
    module: JITModule,
    ctx: codegen::Context,
    func_ids: HashMap<DefId, FuncId>,
}

impl CodeGenerator {
    pub fn new() -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").map_err(|e| {
            BoltError::Codegen { message: format!("Failed to set opt level: {}", e) }
        })?;

        let isa_builder = cranelift_native::builder().map_err(|e| {
            BoltError::Codegen { message: format!("Failed to create ISA builder: {}", e) }
        })?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| BoltError::Codegen { message: format!("Failed to create ISA: {}", e) })?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        let ctx = module.make_context();

        Ok(Self {
            module,
            ctx,
            func_ids: HashMap::new(),
        })
    }

    pub fn compile_crate(&mut self, krate: &Crate) -> Result<()> {
        for (def_id, item) in &krate.items {
            if let ItemKind::Function(f) = &item.kind {
                self.declare_function(*def_id, &item.name, f)?;
            }
        }

        for (def_id, item) in &krate.items {
            if let ItemKind::Function(f) = &item.kind {
                self.compile_function(*def_id, &item.name, f)?;
            }
        }

        self.module.finalize_definitions().map_err(|e| {
            BoltError::Codegen { message: format!("Failed to finalize: {}", e) }
        })?;

        Ok(())
    }

    fn declare_function(&mut self, def_id: DefId, name: &str, func: &Function) -> Result<()> {
        let mut sig = self.module.make_signature();

        for (_, ty) in &func.sig.inputs {
            sig.params.push(AbiParam::new(self.type_to_cl(ty)));
        }

        sig.returns.push(AbiParam::new(self.type_to_cl(&func.sig.output)));

        let func_id = self
            .module
            .declare_function(name, Linkage::Export, &sig)
            .map_err(|e| BoltError::Codegen { message: format!("Failed to declare {}: {}", name, e) })?;

        self.func_ids.insert(def_id, func_id);
        Ok(())
    }

    fn compile_function(&mut self, def_id: DefId, name: &str, func: &Function) -> Result<()> {
        let func_id = self.func_ids[&def_id];

        self.ctx.func.signature = self.module.declarations().get_function_decl(func_id).signature.clone();

        let mut builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut builder_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Get block params before creating translator
        let param_values: Vec<_> = (0..func.sig.inputs.len())
            .map(|i| builder.block_params(entry_block)[i])
            .collect();

        let mut translator = FunctionTranslator::new(builder, &self.func_ids, &self.module);

        for ((param_name, _), val) in func.sig.inputs.iter().zip(param_values) {
            translator.locals.insert(param_name.clone(), val);
        }

        if let Some(ref body) = func.body {
            let result = translator.translate_block(body);
            if let Some(val) = result {
                translator.builder.ins().return_(&[val]);
            } else {
                let zero = translator.builder.ins().iconst(types::I64, 0);
                translator.builder.ins().return_(&[zero]);
            }
        } else {
            let zero = translator.builder.ins().iconst(types::I64, 0);
            translator.builder.ins().return_(&[zero]);
        }

        translator.finalize();

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| BoltError::Codegen { message: format!("Failed to define {}: {}", name, e) })?;

        self.module.clear_context(&mut self.ctx);
        Ok(())
    }

    fn type_to_cl(&self, ty: &HirType) -> types::Type {
        match &ty.kind {
            TypeKind::Unit => types::I64,
            TypeKind::Bool => types::I8,
            TypeKind::Char => types::I32,
            TypeKind::Int(IntType::I8) => types::I8,
            TypeKind::Int(IntType::I16) => types::I16,
            TypeKind::Int(IntType::I32) => types::I32,
            TypeKind::Int(IntType::I64) | TypeKind::Int(IntType::Isize) => types::I64,
            TypeKind::Int(IntType::I128) => types::I128,
            TypeKind::Uint(UintType::U8) => types::I8,
            TypeKind::Uint(UintType::U16) => types::I16,
            TypeKind::Uint(UintType::U32) => types::I32,
            TypeKind::Uint(UintType::U64) | TypeKind::Uint(UintType::Usize) => types::I64,
            TypeKind::Uint(UintType::U128) => types::I128,
            TypeKind::Float(FloatType::F32) => types::F32,
            TypeKind::Float(FloatType::F64) => types::F64,
            TypeKind::Ref { .. } | TypeKind::Ptr { .. } => types::I64,
            _ => types::I64,
        }
    }

    pub fn get_function_ptr(&self, name: &str) -> Option<*const u8> {
        let func_id = self.module.get_name(name)?;
        if let cranelift_module::FuncOrDataId::Func(id) = func_id {
            Some(self.module.get_finalized_function(id))
        } else {
            None
        }
    }

    pub fn run_main(&self) -> Result<i64> {
        let main_ptr = self
            .get_function_ptr("main")
            .ok_or_else(|| BoltError::Codegen { message: "No main function found".to_string() })?;

        let main_fn: fn() -> i64 = unsafe { std::mem::transmute(main_ptr) };
        Ok(main_fn())
    }
}

struct FunctionTranslator<'a, 'b> {
    builder: FunctionBuilder<'b>,
    func_ids: &'a HashMap<DefId, FuncId>,
    module: &'a JITModule,
    locals: IndexMap<String, Value>,
}

impl<'a, 'b> FunctionTranslator<'a, 'b> {
    fn new(
        builder: FunctionBuilder<'b>,
        func_ids: &'a HashMap<DefId, FuncId>,
        module: &'a JITModule,
    ) -> Self {
        Self {
            builder,
            func_ids,
            module,
            locals: IndexMap::new(),
        }
    }

    fn finalize(self) {
        self.builder.finalize();
    }

    fn translate_block(&mut self, block: &HirBlock) -> Option<Value> {
        for stmt in &block.stmts {
            self.translate_stmt(stmt);
        }

        block.expr.as_ref().map(|e| self.translate_expr(e))
    }

    fn translate_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { pattern, init, .. } => {
                if let Some(init) = init {
                    let val = self.translate_expr(init);
                    self.bind_pattern(pattern, val);
                }
            }
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
                self.translate_expr(expr);
            }
            StmtKind::Item(_) => {}
        }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, val: Value) {
        if let PatternKind::Ident { name, .. } = &pattern.kind {
            self.locals.insert(name.clone(), val);
        }
    }

    fn translate_expr(&mut self, expr: &Expr) -> Value {
        match &expr.kind {
            ExprKind::Lit(lit) => self.translate_literal(lit),
            ExprKind::Path(path) => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0].ident;
                    if let Some(&val) = self.locals.get(name) {
                        return val;
                    }
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs_val = self.translate_expr(lhs);
                let rhs_val = self.translate_expr(rhs);
                self.translate_binop(*op, lhs_val, rhs_val)
            }
            ExprKind::Unary { op, expr: inner } => {
                let val = self.translate_expr(inner);
                match op {
                    UnaryOp::Neg => self.builder.ins().ineg(val),
                    UnaryOp::Not => self.builder.ins().bnot(val),
                }
            }
            ExprKind::If { cond, then_branch, else_branch } => {
                let cond_val = self.translate_expr(cond);

                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let merge_block = self.builder.create_block();

                self.builder.append_block_param(merge_block, types::I64);

                self.builder.ins().brif(cond_val, then_block, &[], else_block, &[]);

                self.builder.switch_to_block(then_block);
                self.builder.seal_block(then_block);
                let then_val = self.translate_block(then_branch).unwrap_or_else(|| {
                    self.builder.ins().iconst(types::I64, 0)
                });
                self.builder.ins().jump(merge_block, &[then_val]);

                self.builder.switch_to_block(else_block);
                self.builder.seal_block(else_block);
                let else_val = if let Some(else_expr) = else_branch {
                    self.translate_expr(else_expr)
                } else {
                    self.builder.ins().iconst(types::I64, 0)
                };
                self.builder.ins().jump(merge_block, &[else_val]);

                self.builder.switch_to_block(merge_block);
                self.builder.seal_block(merge_block);

                self.builder.block_params(merge_block)[0]
            }
            ExprKind::Block(block) => {
                self.translate_block(block).unwrap_or_else(|| {
                    self.builder.ins().iconst(types::I64, 0)
                })
            }
            ExprKind::Return(inner) => {
                let val = inner
                    .as_ref()
                    .map(|e| self.translate_expr(e))
                    .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
                self.builder.ins().return_(&[val]);
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Call { func, args } => {
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Tuple(elems) => {
                if elems.is_empty() {
                    self.builder.ins().iconst(types::I64, 0)
                } else {
                    self.translate_expr(&elems[0])
                }
            }
            _ => self.builder.ins().iconst(types::I64, 0),
        }
    }

    fn translate_literal(&mut self, lit: &Literal) -> Value {
        match lit {
            Literal::Bool(b) => self.builder.ins().iconst(types::I8, if *b { 1 } else { 0 }),
            Literal::Char(c) => self.builder.ins().iconst(types::I32, *c as i64),
            Literal::Int(n, _) => self.builder.ins().iconst(types::I64, *n as i64),
            Literal::Uint(n, _) => self.builder.ins().iconst(types::I64, *n as i64),
            Literal::Float(f, Some(FloatType::F32)) => self.builder.ins().f32const(*f as f32),
            Literal::Float(f, _) => self.builder.ins().f64const(*f),
            Literal::Str(_) => self.builder.ins().iconst(types::I64, 0),
            Literal::ByteStr(_) => self.builder.ins().iconst(types::I64, 0),
        }
    }

    fn translate_binop(&mut self, op: BinaryOp, lhs: Value, rhs: Value) -> Value {
        match op {
            BinaryOp::Add => self.builder.ins().iadd(lhs, rhs),
            BinaryOp::Sub => self.builder.ins().isub(lhs, rhs),
            BinaryOp::Mul => self.builder.ins().imul(lhs, rhs),
            BinaryOp::Div => self.builder.ins().sdiv(lhs, rhs),
            BinaryOp::Rem => self.builder.ins().srem(lhs, rhs),
            BinaryOp::BitAnd => self.builder.ins().band(lhs, rhs),
            BinaryOp::BitOr => self.builder.ins().bor(lhs, rhs),
            BinaryOp::BitXor => self.builder.ins().bxor(lhs, rhs),
            BinaryOp::Shl => self.builder.ins().ishl(lhs, rhs),
            BinaryOp::Shr => self.builder.ins().sshr(lhs, rhs),
            BinaryOp::Eq => self.builder.ins().icmp(IntCC::Equal, lhs, rhs),
            BinaryOp::Ne => self.builder.ins().icmp(IntCC::NotEqual, lhs, rhs),
            BinaryOp::Lt => self.builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs),
            BinaryOp::Le => self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs),
            BinaryOp::Gt => self.builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs),
            BinaryOp::Ge => self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs),
            BinaryOp::And | BinaryOp::Or => self.builder.ins().band(lhs, rhs),
        }
    }
}
