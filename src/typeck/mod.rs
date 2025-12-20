//! Type checking for Bolt
//!
//! Performs type inference and checking in parallel per-function.

use crate::error::{BoltError, Diagnostic, DiagnosticEmitter, Result, Span};
use crate::hir::*;
use dashmap::DashMap;
use indexmap::IndexMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

pub type TypeId = u32;

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedType {
    Unit,
    Bool,
    Char,
    Int(IntType),
    Uint(UintType),
    Float(FloatType),
    Str,
    Ref { lifetime: Option<String>, mutable: bool, inner: Box<ResolvedType> },
    Ptr { mutable: bool, inner: Box<ResolvedType> },
    Slice(Box<ResolvedType>),
    Array { elem: Box<ResolvedType>, len: usize },
    Tuple(Vec<ResolvedType>),
    Fn { inputs: Vec<ResolvedType>, output: Box<ResolvedType> },
    Adt { def_id: DefId, args: Vec<ResolvedType> },
    Never,
    Infer(u32),
    Error,
}

pub struct TypeContext {
    types: DashMap<TypeId, ResolvedType>,
    next_type_id: AtomicU32,
    next_infer_id: AtomicU32,
    diagnostics: RwLock<DiagnosticEmitter>,
}

impl TypeContext {
    pub fn new() -> Self {
        Self {
            types: DashMap::new(),
            next_type_id: AtomicU32::new(1),
            next_infer_id: AtomicU32::new(1),
            diagnostics: RwLock::new(DiagnosticEmitter::new()),
        }
    }

    fn alloc_type(&self, ty: ResolvedType) -> TypeId {
        let id = self.next_type_id.fetch_add(1, Ordering::SeqCst);
        self.types.insert(id, ty);
        id
    }

    fn fresh_infer(&self) -> ResolvedType {
        ResolvedType::Infer(self.next_infer_id.fetch_add(1, Ordering::SeqCst))
    }

    pub fn get_type(&self, id: TypeId) -> Option<ResolvedType> {
        self.types.get(&id).map(|t| t.clone())
    }

    pub fn emit_error(&self, diag: Diagnostic) {
        self.diagnostics.write().emit(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.read().has_errors()
    }
}

impl Default for TypeContext {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TypeChecker<'a> {
    ctx: &'a TypeContext,
    krate: &'a Crate,
    local_types: IndexMap<String, ResolvedType>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(ctx: &'a TypeContext, krate: &'a Crate) -> Self {
        Self {
            ctx,
            krate,
            local_types: IndexMap::new(),
        }
    }

    pub fn check_crate(&mut self) -> Result<()> {
        for (_, item) in &self.krate.items {
            self.check_item(item)?;
        }
        Ok(())
    }

    fn check_item(&mut self, item: &Item) -> Result<()> {
        match &item.kind {
            ItemKind::Function(f) => self.check_function(f, &item.name),
            ItemKind::Struct(_) => Ok(()),
            ItemKind::Enum(_) => Ok(()),
            ItemKind::Const(c) => self.check_const(c),
            ItemKind::Static(s) => self.check_static(s),
            ItemKind::Impl(i) => self.check_impl(i),
            ItemKind::Trait(_) => Ok(()),
            ItemKind::TypeAlias(_) => Ok(()),
            ItemKind::Module(_) => Ok(()),
        }
    }

    fn check_function(&mut self, f: &Function, name: &str) -> Result<()> {
        self.local_types.clear();

        for (param_name, param_ty) in &f.sig.inputs {
            let resolved = self.resolve_type(param_ty);
            self.local_types.insert(param_name.clone(), resolved);
        }

        let expected_return = self.resolve_type(&f.sig.output);

        if let Some(ref body) = f.body {
            let actual_return = self.check_block(body)?;
            if !self.types_compatible(&expected_return, &actual_return) {
                self.ctx.emit_error(
                    Diagnostic::error(format!(
                        "Function `{}` returns {:?} but body has type {:?}",
                        name, expected_return, actual_return
                    ))
                    .with_span(body.span),
                );
            }
        }

        Ok(())
    }

    fn check_block(&mut self, block: &Block) -> Result<ResolvedType> {
        for stmt in &block.stmts {
            self.check_stmt(stmt)?;
        }

        if let Some(ref expr) = block.expr {
            self.check_expr(expr)
        } else {
            Ok(ResolvedType::Unit)
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match &stmt.kind {
            StmtKind::Let { pattern, ty, init } => {
                let init_ty = if let Some(init) = init {
                    self.check_expr(init)?
                } else {
                    self.ctx.fresh_infer()
                };

                let declared_ty = ty.as_ref().map(|t| self.resolve_type(t));

                let final_ty = if let Some(declared) = declared_ty {
                    if !self.types_compatible(&declared, &init_ty) {
                        self.ctx.emit_error(
                            Diagnostic::error(format!(
                                "Type mismatch: expected {:?}, found {:?}",
                                declared, init_ty
                            ))
                            .with_span(stmt.span),
                        );
                    }
                    declared
                } else {
                    init_ty
                };

                self.bind_pattern(pattern, &final_ty);
                Ok(())
            }
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
                self.check_expr(expr)?;
                Ok(())
            }
            StmtKind::Item(_) => Ok(()),
        }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, ty: &ResolvedType) {
        match &pattern.kind {
            PatternKind::Ident { name, .. } => {
                self.local_types.insert(name.clone(), ty.clone());
            }
            PatternKind::Tuple(pats) => {
                if let ResolvedType::Tuple(tys) = ty {
                    for (pat, ty) in pats.iter().zip(tys.iter()) {
                        self.bind_pattern(pat, ty);
                    }
                }
            }
            _ => {}
        }
    }

    fn check_expr(&mut self, expr: &Expr) -> Result<ResolvedType> {
        match &expr.kind {
            ExprKind::Lit(lit) => Ok(self.literal_type(lit)),
            ExprKind::Path(path) => self.resolve_path(path, expr.span),
            ExprKind::Unary { op, expr: inner } => {
                let inner_ty = self.check_expr(inner)?;
                match op {
                    UnaryOp::Not => {
                        if matches!(inner_ty, ResolvedType::Bool) {
                            Ok(ResolvedType::Bool)
                        } else {
                            Ok(inner_ty)
                        }
                    }
                    UnaryOp::Neg => Ok(inner_ty),
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs_ty = self.check_expr(lhs)?;
                let rhs_ty = self.check_expr(rhs)?;
                self.check_binop(*op, &lhs_ty, &rhs_ty, expr.span)
            }
            ExprKind::Assign { lhs, rhs } => {
                let lhs_ty = self.check_expr(lhs)?;
                let rhs_ty = self.check_expr(rhs)?;
                if !self.types_compatible(&lhs_ty, &rhs_ty) {
                    self.ctx.emit_error(
                        Diagnostic::error(format!(
                            "Cannot assign {:?} to {:?}",
                            rhs_ty, lhs_ty
                        ))
                        .with_span(expr.span),
                    );
                }
                Ok(ResolvedType::Unit)
            }
            ExprKind::Call { func, args } => {
                let func_ty = self.check_expr(func)?;
                if let ResolvedType::Fn { inputs, output } = func_ty {
                    if args.len() != inputs.len() {
                        self.ctx.emit_error(
                            Diagnostic::error(format!(
                                "Expected {} arguments, got {}",
                                inputs.len(),
                                args.len()
                            ))
                            .with_span(expr.span),
                        );
                    }
                    for (arg, expected) in args.iter().zip(inputs.iter()) {
                        let arg_ty = self.check_expr(arg)?;
                        if !self.types_compatible(expected, &arg_ty) {
                            self.ctx.emit_error(
                                Diagnostic::error(format!(
                                    "Expected {:?}, got {:?}",
                                    expected, arg_ty
                                ))
                                .with_span(arg.span),
                            );
                        }
                    }
                    Ok(*output)
                } else {
                    Ok(self.ctx.fresh_infer())
                }
            }
            ExprKind::If { cond, then_branch, else_branch } => {
                let cond_ty = self.check_expr(cond)?;
                if !matches!(cond_ty, ResolvedType::Bool) {
                    self.ctx.emit_error(
                        Diagnostic::error("If condition must be bool")
                            .with_span(cond.span),
                    );
                }

                let then_ty = self.check_block(then_branch)?;

                if let Some(else_expr) = else_branch {
                    let else_ty = self.check_expr(else_expr)?;
                    if !self.types_compatible(&then_ty, &else_ty) {
                        self.ctx.emit_error(
                            Diagnostic::error(format!(
                                "If branches have incompatible types: {:?} vs {:?}",
                                then_ty, else_ty
                            ))
                            .with_span(expr.span),
                        );
                    }
                    Ok(then_ty)
                } else {
                    Ok(ResolvedType::Unit)
                }
            }
            ExprKind::Block(block) => self.check_block(block),
            ExprKind::Return(inner) => {
                if let Some(inner) = inner {
                    self.check_expr(inner)?;
                }
                Ok(ResolvedType::Never)
            }
            ExprKind::Tuple(elems) => {
                let tys: Vec<_> = elems
                    .iter()
                    .map(|e| self.check_expr(e))
                    .collect::<Result<_>>()?;
                Ok(ResolvedType::Tuple(tys))
            }
            ExprKind::Array(elems) => {
                if elems.is_empty() {
                    Ok(ResolvedType::Array {
                        elem: Box::new(self.ctx.fresh_infer()),
                        len: 0,
                    })
                } else {
                    let first_ty = self.check_expr(&elems[0])?;
                    for elem in &elems[1..] {
                        let ty = self.check_expr(elem)?;
                        if !self.types_compatible(&first_ty, &ty) {
                            self.ctx.emit_error(
                                Diagnostic::error("Array elements must have same type")
                                    .with_span(elem.span),
                            );
                        }
                    }
                    Ok(ResolvedType::Array {
                        elem: Box::new(first_ty),
                        len: elems.len(),
                    })
                }
            }
            ExprKind::Ref { mutable, expr: inner } => {
                let inner_ty = self.check_expr(inner)?;
                Ok(ResolvedType::Ref {
                    lifetime: None,
                    mutable: *mutable,
                    inner: Box::new(inner_ty),
                })
            }
            ExprKind::Deref(inner) => {
                let inner_ty = self.check_expr(inner)?;
                match inner_ty {
                    ResolvedType::Ref { inner, .. } | ResolvedType::Ptr { inner, .. } => Ok(*inner),
                    _ => {
                        self.ctx.emit_error(
                            Diagnostic::error("Cannot dereference non-pointer type")
                                .with_span(expr.span),
                        );
                        Ok(ResolvedType::Error)
                    }
                }
            }
            _ => Ok(self.ctx.fresh_infer()),
        }
    }

    fn literal_type(&self, lit: &Literal) -> ResolvedType {
        match lit {
            Literal::Bool(_) => ResolvedType::Bool,
            Literal::Char(_) => ResolvedType::Char,
            Literal::Int(_, suffix) => {
                suffix.map(ResolvedType::Int).unwrap_or(ResolvedType::Int(IntType::I32))
            }
            Literal::Uint(_, suffix) => {
                suffix.map(ResolvedType::Uint).unwrap_or(ResolvedType::Uint(UintType::U32))
            }
            Literal::Float(_, suffix) => {
                suffix.map(ResolvedType::Float).unwrap_or(ResolvedType::Float(FloatType::F64))
            }
            Literal::Str(_) => ResolvedType::Ref {
                lifetime: Some("static".to_string()),
                mutable: false,
                inner: Box::new(ResolvedType::Str),
            },
            Literal::ByteStr(bytes) => ResolvedType::Ref {
                lifetime: Some("static".to_string()),
                mutable: false,
                inner: Box::new(ResolvedType::Array {
                    elem: Box::new(ResolvedType::Uint(UintType::U8)),
                    len: bytes.len(),
                }),
            },
        }
    }

    fn resolve_path(&self, path: &Path, span: Span) -> Result<ResolvedType> {
        if path.segments.len() == 1 {
            let name = &path.segments[0].ident;
            if let Some(ty) = self.local_types.get(name) {
                return Ok(ty.clone());
            }
        }
        Ok(self.ctx.fresh_infer())
    }

    fn resolve_type(&self, ty: &Type) -> ResolvedType {
        match &ty.kind {
            TypeKind::Unit => ResolvedType::Unit,
            TypeKind::Bool => ResolvedType::Bool,
            TypeKind::Char => ResolvedType::Char,
            TypeKind::Int(i) => ResolvedType::Int(*i),
            TypeKind::Uint(u) => ResolvedType::Uint(*u),
            TypeKind::Float(f) => ResolvedType::Float(*f),
            TypeKind::Str => ResolvedType::Str,
            TypeKind::Ref { lifetime, mutable, inner } => ResolvedType::Ref {
                lifetime: lifetime.clone(),
                mutable: *mutable,
                inner: Box::new(self.resolve_type(inner)),
            },
            TypeKind::Ptr { mutable, inner } => ResolvedType::Ptr {
                mutable: *mutable,
                inner: Box::new(self.resolve_type(inner)),
            },
            TypeKind::Slice(inner) => ResolvedType::Slice(Box::new(self.resolve_type(inner))),
            TypeKind::Array { elem, len } => ResolvedType::Array {
                elem: Box::new(self.resolve_type(elem)),
                len: *len,
            },
            TypeKind::Tuple(elems) => {
                ResolvedType::Tuple(elems.iter().map(|e| self.resolve_type(e)).collect())
            }
            TypeKind::Fn { inputs, output } => ResolvedType::Fn {
                inputs: inputs.iter().map(|i| self.resolve_type(i)).collect(),
                output: Box::new(self.resolve_type(output)),
            },
            TypeKind::Never => ResolvedType::Never,
            TypeKind::Infer => self.ctx.fresh_infer(),
            TypeKind::Path(_) => self.ctx.fresh_infer(),
            TypeKind::Error => ResolvedType::Error,
        }
    }

    fn types_compatible(&self, a: &ResolvedType, b: &ResolvedType) -> bool {
        match (a, b) {
            (ResolvedType::Infer(_), _) | (_, ResolvedType::Infer(_)) => true,
            (ResolvedType::Error, _) | (_, ResolvedType::Error) => true,
            (ResolvedType::Never, _) | (_, ResolvedType::Never) => true,
            (ResolvedType::Unit, ResolvedType::Unit) => true,
            (ResolvedType::Bool, ResolvedType::Bool) => true,
            (ResolvedType::Char, ResolvedType::Char) => true,
            (ResolvedType::Int(a), ResolvedType::Int(b)) => std::mem::discriminant(a) == std::mem::discriminant(b),
            (ResolvedType::Uint(a), ResolvedType::Uint(b)) => std::mem::discriminant(a) == std::mem::discriminant(b),
            (ResolvedType::Float(a), ResolvedType::Float(b)) => std::mem::discriminant(a) == std::mem::discriminant(b),
            (ResolvedType::Str, ResolvedType::Str) => true,
            (ResolvedType::Ref { inner: a, .. }, ResolvedType::Ref { inner: b, .. }) => {
                self.types_compatible(a, b)
            }
            (ResolvedType::Tuple(a), ResolvedType::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| self.types_compatible(a, b))
            }
            _ => false,
        }
    }

    fn check_binop(&self, op: BinaryOp, lhs: &ResolvedType, rhs: &ResolvedType, span: Span) -> Result<ResolvedType> {
        match op {
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                Ok(ResolvedType::Bool)
            }
            BinaryOp::And | BinaryOp::Or => {
                if !matches!(lhs, ResolvedType::Bool) || !matches!(rhs, ResolvedType::Bool) {
                    self.ctx.emit_error(
                        Diagnostic::error("Logical operators require bool operands")
                            .with_span(span),
                    );
                }
                Ok(ResolvedType::Bool)
            }
            _ => {
                if !self.types_compatible(lhs, rhs) {
                    self.ctx.emit_error(
                        Diagnostic::error(format!(
                            "Cannot apply {:?} to {:?} and {:?}",
                            op, lhs, rhs
                        ))
                        .with_span(span),
                    );
                }
                Ok(lhs.clone())
            }
        }
    }

    fn check_const(&mut self, c: &Const) -> Result<()> {
        let expected = self.resolve_type(&c.ty);
        let actual = self.check_expr(&c.value)?;
        if !self.types_compatible(&expected, &actual) {
            self.ctx.emit_error(Diagnostic::error(format!(
                "Const type mismatch: expected {:?}, found {:?}",
                expected, actual
            )));
        }
        Ok(())
    }

    fn check_static(&mut self, s: &Static) -> Result<()> {
        let expected = self.resolve_type(&s.ty);
        let actual = self.check_expr(&s.value)?;
        if !self.types_compatible(&expected, &actual) {
            self.ctx.emit_error(Diagnostic::error(format!(
                "Static type mismatch: expected {:?}, found {:?}",
                expected, actual
            )));
        }
        Ok(())
    }

    fn check_impl(&mut self, _i: &Impl) -> Result<()> {
        Ok(())
    }
}

pub fn check_crate_parallel(ctx: &TypeContext, krate: &Crate) -> Result<()> {
    let functions: Vec<_> = krate
        .items
        .values()
        .filter(|item| matches!(item.kind, ItemKind::Function(_)))
        .collect();

    functions.par_iter().try_for_each(|item| {
        let mut checker = TypeChecker::new(ctx, krate);
        checker.check_item(item)
    })
}
