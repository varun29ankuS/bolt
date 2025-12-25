//! Code generation using Cranelift
//!
//! Compiles HIR to machine code via Cranelift JIT.

use crate::error::{BoltError, Result};
use crate::hir::{
    self, BinaryOp, Block as HirBlock, Crate, DefId, Enum, Expr, ExprKind, FloatType, Function,
    IntType, ItemKind, Literal, Pattern, PatternKind, Stmt, StmtKind, Struct, StructKind,
    Type as HirType, TypeKind, UintType, UnaryOp,
};
use crate::ty::{Ty, TyId, TypeRegistry};
use cranelift::prelude::{
    codegen, settings, types, AbiParam, Block, Configurable, FunctionBuilder, FunctionBuilderContext,
    InstBuilder, IntCC, Value, Variable,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Field information for struct layout
#[derive(Clone)]
struct FieldInfo {
    offset: usize,
    type_name: String,  // Type name for the field (e.g., "i64", "Inner")
    is_struct: bool,    // True if field is a struct type
}

/// Field offset information for a struct
#[derive(Clone)]
struct StructLayout {
    fields: IndexMap<String, FieldInfo>,  // field name -> field info
    size: usize,
}

/// Variant kind for codegen
#[derive(Clone, Copy)]
enum VariantKind {
    Unit,
    Tuple(usize),   // number of fields
    Struct(usize),  // number of fields
}

/// Variant information for an enum
#[derive(Clone)]
struct VariantLayout {
    discriminant: i64,
    kind: VariantKind,
    payload_size: usize,
}

/// Layout information for an enum
#[derive(Clone)]
struct EnumLayout {
    variants: IndexMap<String, VariantLayout>,  // variant name -> layout
    size: usize,  // discriminant (8 bytes) + max payload size
}

/// Trait definition
#[derive(Clone)]
struct TraitDef {
    methods: Vec<String>,  // method names declared in trait
    assoc_types: Vec<String>,  // associated type names declared in trait
}

/// Method implementation info
#[derive(Clone)]
struct MethodImpl {
    func_id: FuncId,
    func_name: String,
}

/// Impl block info
#[derive(Clone)]
struct ImplBlock {
    trait_name: Option<String>,  // None for inherent impl
    methods: HashMap<String, MethodImpl>,  // method name -> implementation
    assoc_types: HashMap<String, String>,  // assoc type name -> concrete type name
}

pub struct CodeGenerator {
    module: JITModule,
    ctx: codegen::Context,
    type_registry: Arc<TypeRegistry>,
    func_ids: HashMap<DefId, FuncId>,
    func_names: HashMap<String, FuncId>,
    struct_layouts: HashMap<String, StructLayout>,
    enum_layouts: HashMap<String, EnumLayout>,
    generic_funcs: HashMap<String, (DefId, Function)>,  // generic function templates
    trait_defs: HashMap<String, TraitDef>,  // trait name -> definition
    type_impls: HashMap<String, Vec<ImplBlock>>,  // type name -> impl blocks
    imports: HashMap<String, String>,  // alias -> full_path from use statements
    /// Functions that have slice parameters - tracks which param indices are slices
    slice_param_funcs: HashMap<String, Vec<usize>>,
    /// Const values: name -> (value as i64, type)
    const_values: HashMap<String, (i64, hir::Type)>,
    /// Static data: name -> (data_id, mutable, type)
    static_data: HashMap<String, (DataId, bool, hir::Type)>,
}

impl CodeGenerator {
    pub fn new(type_registry: Arc<TypeRegistry>) -> Result<Self> {
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

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register runtime functions for heap allocation
        builder.symbol("bolt_malloc", bolt_malloc as *const u8);
        builder.symbol("bolt_free", bolt_free as *const u8);
        builder.symbol("bolt_realloc", bolt_realloc as *const u8);
        builder.symbol("bolt_print_int", bolt_print_int as *const u8);
        builder.symbol("bolt_print_str", bolt_print_str as *const u8);

        // Register Vec runtime functions
        builder.symbol("bolt_vec_new", crate::runtime::bolt_vec_new as *const u8);
        builder.symbol("bolt_vec_with_capacity", crate::runtime::bolt_vec_with_capacity as *const u8);
        builder.symbol("bolt_vec_push", crate::runtime::bolt_vec_push as *const u8);
        builder.symbol("bolt_vec_get", crate::runtime::bolt_vec_get as *const u8);
        builder.symbol("bolt_vec_get_mut", crate::runtime::bolt_vec_get_mut as *const u8);
        builder.symbol("bolt_vec_len", crate::runtime::bolt_vec_len as *const u8);
        builder.symbol("bolt_vec_capacity", crate::runtime::bolt_vec_capacity as *const u8);
        builder.symbol("bolt_vec_is_empty", crate::runtime::bolt_vec_is_empty as *const u8);
        builder.symbol("bolt_vec_pop", crate::runtime::bolt_vec_pop as *const u8);
        builder.symbol("bolt_vec_clear", crate::runtime::bolt_vec_clear as *const u8);
        builder.symbol("bolt_vec_drop", crate::runtime::bolt_vec_drop as *const u8);
        builder.symbol("bolt_vec_from_slice", crate::runtime::bolt_vec_from_slice as *const u8);
        builder.symbol("bolt_vec_iter", crate::runtime::bolt_vec_iter as *const u8);
        builder.symbol("bolt_vec_iter_next", crate::runtime::bolt_vec_iter_next as *const u8);
        builder.symbol("bolt_vec_sum_i64", crate::runtime::bolt_vec_sum_i64 as *const u8);
        builder.symbol("bolt_vec_sum_i32", crate::runtime::bolt_vec_sum_i32 as *const u8);
        builder.symbol("bolt_vec_sum_f64", crate::runtime::bolt_vec_sum_f64 as *const u8);

        // Register String runtime functions
        builder.symbol("bolt_string_new", crate::runtime::bolt_string_new as *const u8);
        builder.symbol("bolt_string_from_utf8", crate::runtime::bolt_string_from_utf8 as *const u8);
        builder.symbol("bolt_string_len", crate::runtime::bolt_string_len as *const u8);
        builder.symbol("bolt_string_as_ptr", crate::runtime::bolt_string_as_ptr as *const u8);
        builder.symbol("bolt_string_push_byte", crate::runtime::bolt_string_push_byte as *const u8);
        builder.symbol("bolt_string_push_str", crate::runtime::bolt_string_push_str as *const u8);
        builder.symbol("bolt_string_drop", crate::runtime::bolt_string_drop as *const u8);

        // Register HashMap runtime functions
        builder.symbol("bolt_hashmap_new", crate::runtime::bolt_hashmap_new as *const u8);
        builder.symbol("bolt_hashmap_with_capacity", crate::runtime::bolt_hashmap_with_capacity as *const u8);
        builder.symbol("bolt_hashmap_insert", crate::runtime::bolt_hashmap_insert as *const u8);
        builder.symbol("bolt_hashmap_get", crate::runtime::bolt_hashmap_get as *const u8);
        builder.symbol("bolt_hashmap_contains_key", crate::runtime::bolt_hashmap_contains_key as *const u8);
        builder.symbol("bolt_hashmap_remove", crate::runtime::bolt_hashmap_remove as *const u8);
        builder.symbol("bolt_hashmap_len", crate::runtime::bolt_hashmap_len as *const u8);
        builder.symbol("bolt_hashmap_is_empty", crate::runtime::bolt_hashmap_is_empty as *const u8);
        builder.symbol("bolt_hashmap_clear", crate::runtime::bolt_hashmap_clear as *const u8);
        builder.symbol("bolt_hashmap_drop", crate::runtime::bolt_hashmap_drop as *const u8);

        // Register Box runtime functions
        builder.symbol("bolt_box_new", crate::runtime::bolt_box_new as *const u8);
        builder.symbol("bolt_box_drop", crate::runtime::bolt_box_drop as *const u8);

        // Register I/O functions
        builder.symbol("bolt_print_i64", crate::runtime::bolt_print_i64 as *const u8);
        builder.symbol("bolt_print_u64", crate::runtime::bolt_print_u64 as *const u8);
        builder.symbol("bolt_print_f64", crate::runtime::bolt_print_f64 as *const u8);
        builder.symbol("bolt_print_newline", crate::runtime::bolt_print_newline as *const u8);
        builder.symbol("bolt_print_bool", crate::runtime::bolt_print_bool as *const u8);
        builder.symbol("bolt_print_char", crate::runtime::bolt_print_char as *const u8);

        // Register panic functions
        builder.symbol("bolt_panic", crate::runtime::bolt_panic as *const u8);
        builder.symbol("bolt_abort", crate::runtime::bolt_abort as *const u8);

        let module = JITModule::new(builder);
        let ctx = module.make_context();

        Ok(Self {
            module,
            ctx,
            type_registry,
            func_ids: HashMap::new(),
            func_names: HashMap::new(),
            struct_layouts: HashMap::new(),
            enum_layouts: HashMap::new(),
            generic_funcs: HashMap::new(),
            trait_defs: HashMap::new(),
            type_impls: HashMap::new(),
            imports: HashMap::new(),
            slice_param_funcs: HashMap::new(),
            const_values: HashMap::new(),
            static_data: HashMap::new(),
        })
    }

    pub fn compile_crate(&mut self, krate: &Crate) -> Result<()> {
        // Copy imports from crate
        self.imports = krate.imports.clone();

        // Declare runtime functions
        self.declare_runtime_functions()?;

        // Collect struct layouts
        for (_, item) in &krate.items {
            if let ItemKind::Struct(s) = &item.kind {
                self.register_struct(&item.name, s);
            }
        }

        // Collect enum layouts
        for (_, item) in &krate.items {
            if let ItemKind::Enum(e) = &item.kind {
                self.register_enum(&item.name, e);
            }
        }

        // Collect trait definitions
        for (_, item) in &krate.items {
            if let ItemKind::Trait(t) = &item.kind {
                self.register_trait(&item.name, t);
            }
        }

        // Collect const values
        for (_, item) in &krate.items {
            if let ItemKind::Const(c) = &item.kind {
                self.register_const(&item.name, c);
            }
        }

        // Collect static items
        for (_, item) in &krate.items {
            if let ItemKind::Static(s) = &item.kind {
                self.register_static(&item.name, s)?;
            }
        }

        // Declare all functions first (generic ones treat type params as i64)
        for (def_id, item) in &krate.items {
            if let ItemKind::Function(f) = &item.kind {
                if f.sig.generics.params.iter().any(|p| matches!(p, hir::GenericParam::Type { .. })) {
                    // Store generic function template for future proper monomorphization
                    self.generic_funcs.insert(item.name.clone(), (*def_id, f.clone()));
                }
                // Declare all functions - generic params treated as i64
                self.declare_function(*def_id, &item.name, f)?;
            }
        }

        // Collect impl blocks AFTER functions are declared so we have func_ids
        for (_, item) in &krate.items {
            if let ItemKind::Impl(i) = &item.kind {
                self.register_impl(i, krate);
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

    fn declare_runtime_functions(&mut self) -> Result<()> {
        // Helper to declare a function
        macro_rules! declare_fn {
            ($name:expr, [$($param:expr),*], [$($ret:expr),*]) => {{
                let mut sig = self.module.make_signature();
                $(sig.params.push(AbiParam::new($param));)*
                $(sig.returns.push(AbiParam::new($ret));)*
                let id = self.module
                    .declare_function($name, Linkage::Import, &sig)
                    .map_err(|e| BoltError::Codegen { message: format!("Failed to declare {}: {}", $name, e) })?;
                self.func_names.insert($name.to_string(), id);
            }};
        }

        // Basic allocation
        declare_fn!("bolt_malloc", [types::I64], [types::I64]);
        declare_fn!("bolt_free", [types::I64, types::I64], []);
        declare_fn!("bolt_realloc", [types::I64, types::I64, types::I64], [types::I64]);

        // Legacy print functions
        declare_fn!("bolt_print_int", [types::I64], []);
        declare_fn!("bolt_print_str", [types::I64, types::I64], []);

        // Vec functions
        // bolt_vec_new() -> (ptr, len, cap) as 3 i64s returned via struct return
        // For simplicity, we return ptr to stack-allocated struct
        declare_fn!("bolt_vec_new", [], [types::I64, types::I64, types::I64]);
        declare_fn!("bolt_vec_with_capacity", [types::I64, types::I64], [types::I64, types::I64, types::I64]);
        declare_fn!("bolt_vec_push", [types::I64, types::I64, types::I64], []);
        declare_fn!("bolt_vec_get", [types::I64, types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_vec_get_mut", [types::I64, types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_vec_len", [types::I64], [types::I64]);
        declare_fn!("bolt_vec_capacity", [types::I64], [types::I64]);
        declare_fn!("bolt_vec_is_empty", [types::I64], [types::I8]);
        declare_fn!("bolt_vec_pop", [types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_vec_clear", [types::I64], []);
        declare_fn!("bolt_vec_drop", [types::I64, types::I64], []);
        declare_fn!("bolt_vec_from_slice", [types::I64, types::I64, types::I64], [types::I64, types::I64, types::I64]);
        declare_fn!("bolt_vec_sum_i64", [types::I64], [types::I64]);
        declare_fn!("bolt_vec_sum_i32", [types::I64], [types::I32]);
        declare_fn!("bolt_vec_sum_f64", [types::I64], [types::F64]);

        // String functions
        declare_fn!("bolt_string_new", [], [types::I64, types::I64, types::I64]);
        declare_fn!("bolt_string_from_utf8", [types::I64, types::I64], [types::I64, types::I64, types::I64]);
        declare_fn!("bolt_string_len", [types::I64], [types::I64]);
        declare_fn!("bolt_string_as_ptr", [types::I64], [types::I64]);
        declare_fn!("bolt_string_push_byte", [types::I64, types::I8], []);
        declare_fn!("bolt_string_push_str", [types::I64, types::I64, types::I64], []);
        declare_fn!("bolt_string_drop", [types::I64], []);

        // HashMap functions
        // HashMap layout: entries_ptr(8) + len(8) + cap(8) = 24 bytes
        declare_fn!("bolt_hashmap_new", [], [types::I64, types::I64, types::I64]);
        declare_fn!("bolt_hashmap_with_capacity", [types::I64], [types::I64, types::I64, types::I64]);
        declare_fn!("bolt_hashmap_insert", [types::I64, types::I64, types::I64], []);
        declare_fn!("bolt_hashmap_get", [types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_hashmap_contains_key", [types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_hashmap_remove", [types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_hashmap_len", [types::I64], [types::I64]);
        declare_fn!("bolt_hashmap_is_empty", [types::I64], [types::I64]);
        declare_fn!("bolt_hashmap_clear", [types::I64], []);
        declare_fn!("bolt_hashmap_drop", [types::I64], []);

        // Slice functions
        // Slices are fat pointers: (ptr: i64, len: i64) = 16 bytes
        declare_fn!("bolt_slice_len", [types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_slice_get", [types::I64, types::I64, types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_slice_get_unchecked", [types::I64, types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_slice_bounds_check", [types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_slice_is_empty", [types::I64], [types::I64]);
        declare_fn!("bolt_slice_first", [types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_slice_last", [types::I64, types::I64, types::I64], [types::I64]);

        // Box functions
        declare_fn!("bolt_box_new", [types::I64, types::I64], [types::I64]);
        declare_fn!("bolt_box_drop", [types::I64, types::I64, types::I64], []);

        // I/O functions
        declare_fn!("bolt_print_i64", [types::I64], []);
        declare_fn!("bolt_print_u64", [types::I64], []);
        declare_fn!("bolt_print_f64", [types::F64], []);
        declare_fn!("bolt_print_newline", [], []);
        declare_fn!("bolt_print_bool", [types::I8], []);
        declare_fn!("bolt_print_char", [types::I32], []);

        // Panic/abort
        declare_fn!("bolt_panic", [types::I64, types::I64], []);
        declare_fn!("bolt_abort", [], []);

        Ok(())
    }

    fn register_trait(&mut self, name: &str, t: &hir::Trait) {
        // Extract associated type names from the trait
        let assoc_types: Vec<String> = t.assoc_types
            .iter()
            .map(|at| at.name.clone())
            .collect();

        self.trait_defs.insert(name.to_string(), TraitDef {
            methods: Vec::new(),
            assoc_types,
        });
    }

    /// Register a const item - evaluate the expression at compile time
    fn register_const(&mut self, name: &str, c: &hir::Const) {
        // Try to evaluate the const expression at compile time
        if let Some(value) = self.eval_const_expr(&c.value) {
            self.const_values.insert(name.to_string(), (value, c.ty.clone()));
        }
    }

    /// Evaluate a constant expression at compile time
    fn eval_const_expr(&self, expr: &Expr) -> Option<i64> {
        match &expr.kind {
            ExprKind::Lit(lit) => match lit {
                Literal::Int(v, _) => Some(*v as i64),
                Literal::Uint(v, _) => Some(*v as i64),
                Literal::Bool(b) => Some(if *b { 1 } else { 0 }),
                Literal::Char(c) => Some(*c as i64),
                _ => None,
            },
            ExprKind::Unary { op, expr } => {
                let val = self.eval_const_expr(expr)?;
                match op {
                    UnaryOp::Neg => Some(-val),
                    UnaryOp::Not => Some(!val),
                    _ => None,
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.eval_const_expr(lhs)?;
                let r = self.eval_const_expr(rhs)?;
                match op {
                    BinaryOp::Add => Some(l + r),
                    BinaryOp::Sub => Some(l - r),
                    BinaryOp::Mul => Some(l * r),
                    BinaryOp::Div => if r != 0 { Some(l / r) } else { None },
                    BinaryOp::Rem => if r != 0 { Some(l % r) } else { None },
                    BinaryOp::BitAnd => Some(l & r),
                    BinaryOp::BitOr => Some(l | r),
                    BinaryOp::BitXor => Some(l ^ r),
                    BinaryOp::Shl => Some(l << r),
                    BinaryOp::Shr => Some(l >> r),
                    _ => None,
                }
            }
            ExprKind::Path(path) => {
                // Look up other const values
                if path.segments.len() == 1 {
                    let name = &path.segments[0].ident;
                    self.const_values.get(name).map(|(v, _)| *v)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Register a static item - create a data segment for it
    fn register_static(&mut self, name: &str, s: &hir::Static) -> Result<()> {
        // Evaluate initial value
        let value = self.eval_const_expr(&s.value).unwrap_or(0);

        // Determine size based on type
        let size = self.type_size(&s.ty);

        // Create initial data
        let mut data = vec![0u8; size];
        // Write value as little-endian
        for i in 0..std::cmp::min(8, size) {
            data[i] = ((value >> (i * 8)) & 0xff) as u8;
        }

        // Create data description with the initial value
        let mut data_desc = DataDescription::new();
        data_desc.define(data.into_boxed_slice());

        // Declare data
        let data_id = self.module.declare_data(
            name,
            if s.mutable { Linkage::Export } else { Linkage::Local },
            s.mutable,
            false,
        ).map_err(|e| BoltError::Codegen { message: format!("Failed to declare static {}: {}", name, e) })?;

        // Define the data
        self.module.define_data(data_id, &data_desc)
            .map_err(|e| BoltError::Codegen { message: format!("Failed to define static {}: {}", name, e) })?;

        self.static_data.insert(name.to_string(), (data_id, s.mutable, s.ty.clone()));
        Ok(())
    }

    /// Get size of a type in bytes
    fn type_size(&self, ty: &HirType) -> usize {
        match &ty.kind {
            TypeKind::Bool => 1,
            TypeKind::Char => 4,
            TypeKind::Int(int_ty) => match int_ty {
                IntType::I8 => 1,
                IntType::I16 => 2,
                IntType::I32 => 4,
                IntType::I64 | IntType::Isize => 8,
                IntType::I128 => 16,
            },
            TypeKind::Uint(uint_ty) => match uint_ty {
                UintType::U8 => 1,
                UintType::U16 => 2,
                UintType::U32 => 4,
                UintType::U64 | UintType::Usize => 8,
                UintType::U128 => 16,
            },
            TypeKind::Float(float_ty) => match float_ty {
                FloatType::F32 => 4,
                FloatType::F64 => 8,
            },
            TypeKind::Ptr { .. } | TypeKind::Ref { .. } => 8,
            TypeKind::Path(path) => {
                // Check if it's a struct
                let name = path.segments.last().map(|s| s.ident.as_str()).unwrap_or("");
                if let Some(layout) = self.struct_layouts.get(name) {
                    layout.size
                } else {
                    8 // Default pointer size
                }
            }
            _ => 8, // Default to 8 bytes
        }
    }

    fn register_impl(&mut self, i: &hir::Impl, krate: &Crate) {
        // Get the type name from self_ty
        let type_name = self.type_to_name(&i.self_ty);
        if type_name.is_empty() {
            return;
        }

        // Get trait name if this is a trait impl
        let trait_name = i.trait_ref.as_ref().map(|path| {
            path.segments.last().map(|s| s.ident.clone()).unwrap_or_default()
        });

        // Build method map from actual DefIds in the impl block
        let mut methods = HashMap::new();
        for &method_def_id in &i.items {
            if let Some(method_item) = krate.items.get(&method_def_id) {
                if let ItemKind::Function(_) = &method_item.kind {
                    let method_name = method_item.name.clone();
                    if let Some(&func_id) = self.func_ids.get(&method_def_id) {
                        methods.insert(method_name, MethodImpl {
                            func_id,
                            func_name: method_item.name.clone(),
                        });
                    }
                }
            }
        }

        // Fallback: scan for Type_method naming convention for compatibility
        let prefix = format!("{}_", type_name);
        for (_, item) in &krate.items {
            if let ItemKind::Function(_) = &item.kind {
                if item.name.starts_with(&prefix) {
                    let method_name = item.name.strip_prefix(&prefix)
                        .unwrap_or(&item.name)
                        .to_string();
                    if !methods.contains_key(&method_name) {
                        if let Some(&func_id) = self.func_ids.get(&item.id) {
                            methods.insert(method_name, MethodImpl {
                                func_id,
                                func_name: item.name.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Extract associated type bindings
        let mut assoc_types = HashMap::new();
        for binding in &i.assoc_types {
            let concrete_type_name = self.type_to_name(&binding.ty);
            assoc_types.insert(binding.name.clone(), concrete_type_name);
        }

        // Add impl block to type's impl list
        self.type_impls
            .entry(type_name)
            .or_insert_with(Vec::new)
            .push(ImplBlock {
                trait_name,
                methods,
                assoc_types,
            });
    }

    /// Check if a type is &[T] (reference to a slice)
    fn is_slice_ref(ty: &HirType) -> bool {
        if let TypeKind::Ref { inner, .. } = &ty.kind {
            matches!(inner.kind, TypeKind::Slice(_))
        } else {
            false
        }
    }

    /// Check if a type is &[T; N] (reference to an array) - returns the length
    fn is_array_ref(ty: &HirType) -> Option<usize> {
        if let TypeKind::Ref { inner, .. } = &ty.kind {
            if let TypeKind::Array { len, .. } = &inner.kind {
                return Some(*len);
            }
        }
        None
    }

    fn type_to_name(&self, ty: &HirType) -> String {
        match &ty.kind {
            TypeKind::Path(path) => {
                // Join all segments for module-qualified types (e.g., mymod::Point)
                path.segments.iter()
                    .map(|s| s.ident.as_str())
                    .collect::<Vec<_>>()
                    .join("::")
            }
            TypeKind::Int(int_ty) => {
                use crate::hir::IntType;
                match int_ty {
                    IntType::I8 => "i8".to_string(),
                    IntType::I16 => "i16".to_string(),
                    IntType::I32 => "i32".to_string(),
                    IntType::I64 => "i64".to_string(),
                    IntType::I128 => "i128".to_string(),
                    IntType::Isize => "isize".to_string(),
                }
            }
            TypeKind::Uint(uint_ty) => {
                use crate::hir::UintType;
                match uint_ty {
                    UintType::U8 => "u8".to_string(),
                    UintType::U16 => "u16".to_string(),
                    UintType::U32 => "u32".to_string(),
                    UintType::U64 => "u64".to_string(),
                    UintType::U128 => "u128".to_string(),
                    UintType::Usize => "usize".to_string(),
                }
            }
            TypeKind::Bool => "bool".to_string(),
            TypeKind::Str => "str".to_string(),
            TypeKind::Ref { inner, .. } => {
                // For references, return the inner type name
                self.type_to_name(inner)
            }
            _ => String::new(),
        }
    }

    fn register_struct(&mut self, name: &str, s: &Struct) {
        let mut fields = IndexMap::new();
        let mut offset = 0usize;

        if let StructKind::Named(ref field_list) = s.kind {
            for field in field_list {
                // Get field type name
                let type_name = self.type_to_name(&field.ty);
                // Check if field is a struct type
                let is_struct = self.struct_layouts.contains_key(&type_name);
                // Get actual field size - nested structs use their full size
                let field_size = self.get_type_size(&field.ty);

                fields.insert(field.name.clone(), FieldInfo {
                    offset,
                    type_name,
                    is_struct,
                });
                offset += field_size;
            }
        }

        self.struct_layouts.insert(name.to_string(), StructLayout { fields, size: offset });
    }

    /// Get the size of a type in bytes
    fn get_type_size(&self, ty: &HirType) -> usize {
        match &ty.kind {
            TypeKind::Path(path) => {
                if let Some(seg) = path.segments.first() {
                    let type_name = &seg.ident;
                    // Check if it's a known struct type
                    if let Some(layout) = self.struct_layouts.get(type_name) {
                        return layout.size;
                    }
                    // Primitives and references are 8 bytes
                    match type_name.as_str() {
                        "i64" | "u64" | "f64" | "isize" | "usize" | "bool" => 8,
                        "i32" | "u32" | "f32" => 8,  // Align to 8 for simplicity
                        "i16" | "u16" => 8,
                        "i8" | "u8" => 8,
                        _ => 8,  // Default to pointer size
                    }
                } else {
                    8
                }
            }
            TypeKind::Ref { .. } => 8,  // References are pointer-sized
            TypeKind::Slice(_) | TypeKind::Array { .. } => 16,  // ptr + len
            TypeKind::Tuple(elements) => elements.len() * 8,
            _ => 8,
        }
    }

    fn register_enum(&mut self, name: &str, e: &Enum) {
        let mut variants = IndexMap::new();
        let mut max_payload = 0usize;

        for (disc, variant) in e.variants.iter().enumerate() {
            let (kind, payload_size) = match &variant.kind {
                StructKind::Unit => (VariantKind::Unit, 0),
                StructKind::Tuple(types) => (VariantKind::Tuple(types.len()), types.len() * 8),
                StructKind::Named(fields) => (VariantKind::Struct(fields.len()), fields.len() * 8),
            };
            max_payload = max_payload.max(payload_size);

            variants.insert(
                variant.name.clone(),
                VariantLayout {
                    discriminant: disc as i64,
                    kind,
                    payload_size,
                },
            );
        }

        // Size = 8 bytes discriminant + max payload
        let size = 8 + max_payload;
        self.enum_layouts.insert(name.to_string(), EnumLayout { variants, size });
    }

    fn declare_function(&mut self, def_id: DefId, name: &str, func: &Function) -> Result<()> {
        let mut sig = self.module.make_signature();

        // Extract module prefix from function name (e.g., "mymod::Point_new" -> "mymod")
        let module_prefix = Self::extract_module_prefix(name);

        // If function returns a struct, use sret convention:
        // Add hidden first parameter for caller-allocated return space
        let returns_struct = self.is_struct_type_with_prefix(&func.sig.output, &module_prefix);
        if returns_struct {
            sig.params.push(AbiParam::new(types::I64)); // sret pointer
        }

        // Track which parameters are slice references (need 2 values: ptr + len)
        let mut slice_params = Vec::new();
        for (i, (_, ty)) in func.sig.inputs.iter().enumerate() {
            if Self::is_slice_ref(ty) {
                slice_params.push(i);
                // Slice: add ptr + len as two separate params
                sig.params.push(AbiParam::new(types::I64)); // ptr
                sig.params.push(AbiParam::new(types::I64)); // len
            } else {
                sig.params.push(AbiParam::new(self.type_to_cl(ty)));
            }
        }

        // Store slice param info for use during call translation
        if !slice_params.is_empty() {
            self.slice_param_funcs.insert(name.to_string(), slice_params);
        }

        // Still return the pointer (for convenience in chaining)
        sig.returns.push(AbiParam::new(types::I64));

        let func_id = self
            .module
            .declare_function(name, Linkage::Export, &sig)
            .map_err(|e| BoltError::Codegen { message: format!("Failed to declare {}: {}", name, e) })?;

        self.func_ids.insert(def_id, func_id);
        self.func_names.insert(name.to_string(), func_id);
        Ok(())
    }

    fn compile_function(&mut self, def_id: DefId, name: &str, func: &Function) -> Result<()> {
        let func_id = self.func_ids[&def_id];
        let module_prefix = Self::extract_module_prefix(name);
        let returns_struct = self.is_struct_type_with_prefix(&func.sig.output, &module_prefix);

        self.ctx.func.signature = self.module.declarations().get_function_decl(func_id).signature.clone();

        // Pre-compute Cranelift types and type names for parameters before creating builder
        let param_type_names: Vec<String> = func.sig.inputs.iter()
            .map(|(_, ty)| self.type_to_name(ty))
            .collect();

        // Identify which parameters are slices
        let slice_param_indices: Vec<usize> = func.sig.inputs.iter()
            .enumerate()
            .filter(|(_, (_, ty))| Self::is_slice_ref(ty))
            .map(|(i, _)| i)
            .collect();

        // Pre-compute Cranelift types for each parameter (before borrowing ctx.func)
        let param_cl_types: Vec<types::Type> = func.sig.inputs.iter()
            .map(|(_, ty)| self.type_to_cl(ty))
            .collect();

        let mut builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut builder_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Handle sret parameter if function returns a struct
        let sret_ptr = if returns_struct {
            Some(builder.block_params(entry_block)[0])
        } else {
            None
        };

        // Copy block params into a Vec to avoid borrow issues
        let block_params: Vec<Value> = builder.block_params(entry_block).to_vec();
        let mut block_param_idx = if returns_struct { 1 } else { 0 };

        // Extract module prefix from function name (e.g., "outer::get_inner" -> "outer")
        let current_module = if let Some(pos) = name.rfind("::") {
            name[..pos].to_string()
        } else {
            String::new()
        };

        let mut translator = FunctionTranslator::new(builder, &self.type_registry, &self.func_ids, &self.func_names, &mut self.module, &self.struct_layouts, &self.enum_layouts, &self.type_impls, &self.imports, &self.slice_param_funcs, &self.const_values, &self.static_data, current_module);
        translator.sret_ptr = sret_ptr;

        for (i, ((param_name, _ty), type_name)) in func.sig.inputs.iter().zip(param_type_names.into_iter()).enumerate() {
            if slice_param_indices.contains(&i) {
                // Slice parameter: get ptr and len as two block params
                let ptr_val = block_params[block_param_idx];
                let len_val = block_params[block_param_idx + 1];
                block_param_idx += 2;

                // Store in a stack slot as a fat pointer (16 bytes)
                let slot = translator.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                    16,
                    0,
                ));
                let slot_addr = translator.builder.ins().stack_addr(types::I64, slot, 0);
                translator.builder.ins().store(cranelift::prelude::MemFlags::new(), ptr_val, slot_addr, 0);
                translator.builder.ins().store(cranelift::prelude::MemFlags::new(), len_val, slot_addr, 8);

                // Store pointer to fat pointer as the local
                let var = Variable::from_u32(translator.next_var as u32);
                translator.next_var += 1;
                translator.builder.declare_var(var, types::I64);
                translator.builder.def_var(var, slot_addr);
                translator.locals.insert(param_name.clone(), LocalInfo { var, ty: types::I64 });

                // Also store ptr and len separately for direct access
                let ptr_var = Variable::from_u32(translator.next_var as u32);
                translator.next_var += 1;
                translator.builder.declare_var(ptr_var, types::I64);
                translator.builder.def_var(ptr_var, ptr_val);
                translator.locals.insert(format!("{}_ptr", param_name), LocalInfo { var: ptr_var, ty: types::I64 });

                let len_var = Variable::from_u32(translator.next_var as u32);
                translator.next_var += 1;
                translator.builder.declare_var(len_var, types::I64);
                translator.builder.def_var(len_var, len_val);
                translator.locals.insert(format!("{}_len", param_name), LocalInfo { var: len_var, ty: types::I64 });

                if !type_name.is_empty() {
                    translator.local_types.insert(param_name.clone(), type_name);
                }
            } else {
                // Regular parameter
                let val = block_params[block_param_idx];
                block_param_idx += 1;

                let cl_ty = param_cl_types[i];
                let var = Variable::from_u32(translator.next_var as u32);
                translator.next_var += 1;
                translator.builder.declare_var(var, cl_ty);
                translator.builder.def_var(var, val);
                translator.locals.insert(param_name.clone(), LocalInfo { var, ty: cl_ty });
                if !type_name.is_empty() {
                    translator.local_types.insert(param_name.clone(), type_name);
                }
            }
        }

        // Save sret_ptr before translation (struct expressions will clear it after use)
        let original_sret = translator.sret_ptr;

        if let Some(ref body) = func.body {
            let result = translator.translate_block(body);
            if let Some(val) = result {
                // If this function returns a struct, return the sret pointer
                if let Some(sret) = original_sret {
                    // The struct data is already in sret (filled by struct expression)
                    translator.builder.ins().return_(&[sret]);
                } else {
                    // Coerce return value to i64 (function signature always uses i64)
                    let val = translator.coerce_to_type(val, types::I64);
                    translator.builder.ins().return_(&[val]);
                }
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
            .map_err(|e| BoltError::Codegen { message: format!("Failed to define {}: {:?}", name, e) })?;

        self.module.clear_context(&mut self.ctx);
        Ok(())
    }

    /// Check if a type is a struct (returns by pointer)
    /// module_prefix: Optional module prefix to try (e.g., "mymod" for types in module mymod)
    fn is_struct_type_with_prefix(&self, ty: &HirType, module_prefix: &str) -> bool {
        if let TypeKind::Path(path) = &ty.kind {
            let type_name: String = path.segments.iter()
                .map(|s| s.ident.as_str())
                .collect::<Vec<_>>()
                .join("::");

            // Try exact match first
            if self.struct_layouts.contains_key(&type_name) {
                return true;
            }

            // Try with module prefix
            if !module_prefix.is_empty() {
                let qualified_name = format!("{}::{}", module_prefix, type_name);
                if self.struct_layouts.contains_key(&qualified_name) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if a type is a struct (returns by pointer) - no module context
    fn is_struct_type(&self, ty: &HirType) -> bool {
        self.is_struct_type_with_prefix(ty, "")
    }

    /// Extract module prefix from function name
    /// e.g., "mymod::Point_new" -> "mymod", "outer::inner::Foo_bar" -> "outer::inner", "Point_new" -> ""
    fn extract_module_prefix(func_name: &str) -> String {
        // Method names follow pattern: module::Type_method or Type_method
        // Find the last _ to get the type part, then extract module from that
        if let Some(underscore_pos) = func_name.rfind('_') {
            let type_part = &func_name[..underscore_pos];
            // Now find the module prefix from type_part (e.g., "mymod::Point" -> "mymod")
            if let Some(last_colon) = type_part.rfind("::") {
                return type_part[..last_colon].to_string();
            }
        } else if let Some(last_colon) = func_name.rfind("::") {
            // Plain function in module (e.g., "mymod::func")
            return func_name[..last_colon].to_string();
        }
        String::new()
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
            TypeKind::Ref { .. } | TypeKind::Ptr { .. } => types::I64, // Pointers are always i64
            TypeKind::Path(path) => {
                // Resolve the path type using the type registry
                if let Some(seg) = path.segments.first() {
                    let name = &seg.ident;
                    // Check if it's a struct - structs are passed by pointer
                    if self.type_registry.is_struct(name) || self.struct_layouts.contains_key(name) {
                        return types::I64; // Struct pointer
                    }
                    // Check if it's an enum - enums are passed by pointer
                    if self.type_registry.is_enum(name) || self.enum_layouts.contains_key(name) {
                        return types::I64; // Enum pointer
                    }
                    // Resolve via type registry
                    let ty_id = self.type_registry.resolve_hir_type(ty);
                    return self.ty_id_to_cl(ty_id);
                }
                types::I64
            }
            TypeKind::Tuple(_) => types::I64, // Tuples passed by pointer
            TypeKind::Array { .. } => types::I64, // Arrays passed by pointer
            TypeKind::Slice(_) => types::I64, // Slices are fat pointers
            TypeKind::Fn { .. } => types::I64, // Function pointers
            _ => types::I64,
        }
    }

    /// Convert a resolved TyId to a Cranelift type
    fn ty_id_to_cl(&self, ty_id: TyId) -> types::Type {
        match self.type_registry.get(ty_id) {
            Some(Ty::Unit) => types::I64,
            Some(Ty::Bool) => types::I8,
            Some(Ty::Char) => types::I32,
            Some(Ty::Int(IntType::I8)) => types::I8,
            Some(Ty::Int(IntType::I16)) => types::I16,
            Some(Ty::Int(IntType::I32)) => types::I32,
            Some(Ty::Int(IntType::I64)) | Some(Ty::Int(IntType::Isize)) => types::I64,
            Some(Ty::Int(IntType::I128)) => types::I128,
            Some(Ty::Uint(UintType::U8)) => types::I8,
            Some(Ty::Uint(UintType::U16)) => types::I16,
            Some(Ty::Uint(UintType::U32)) => types::I32,
            Some(Ty::Uint(UintType::U64)) | Some(Ty::Uint(UintType::Usize)) => types::I64,
            Some(Ty::Uint(UintType::U128)) => types::I128,
            Some(Ty::Float(FloatType::F32)) => types::F32,
            Some(Ty::Float(FloatType::F64)) => types::F64,
            Some(Ty::Ref { .. }) | Some(Ty::Ptr { .. }) => types::I64,
            Some(Ty::Adt { .. }) => types::I64, // ADTs passed by pointer
            Some(Ty::Tuple(_)) => types::I64,
            Some(Ty::Array { .. }) => types::I64,
            Some(Ty::Slice(_)) => types::I64,
            Some(Ty::Fn { .. }) => types::I64,
            Some(Ty::Str) => types::I64, // &str is a fat pointer
            Some(Ty::Param { .. }) => types::I64, // Type params treated as i64 until monomorphized
            Some(Ty::ImplTrait { concrete: Some(ty), .. }) => self.ty_id_to_cl(ty), // Resolve to concrete type
            Some(Ty::ImplTrait { .. }) => types::I64, // Unresolved impl Trait
            Some(Ty::DynTrait { .. }) => types::I64, // dyn Trait is a fat pointer (data + vtable)
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

/// Stored closure definition for deferred evaluation
#[derive(Clone)]
struct ClosureDef {
    params: Vec<(Pattern, Option<hir::Type>)>,
    body: Box<Expr>,
}

/// Tracked type information for a local variable
#[derive(Clone, Copy)]
struct LocalInfo {
    var: Variable,
    ty: types::Type,
}

struct FunctionTranslator<'a, 'b> {
    builder: FunctionBuilder<'b>,
    type_registry: &'a TypeRegistry,
    func_ids: &'a HashMap<DefId, FuncId>,
    func_names: &'a HashMap<String, FuncId>,
    module: &'a mut JITModule,
    struct_layouts: &'a HashMap<String, StructLayout>,
    enum_layouts: &'a HashMap<String, EnumLayout>,
    type_impls: &'a HashMap<String, Vec<ImplBlock>>,
    imports: &'a HashMap<String, String>,  // alias -> full_path from use statements
    slice_param_funcs: &'a HashMap<String, Vec<usize>>,  // func name -> slice param indices
    const_values: &'a HashMap<String, (i64, hir::Type)>,  // const name -> (value, type)
    static_data: &'a HashMap<String, (DataId, bool, hir::Type)>,  // static name -> (data_id, mutable, type)
    locals: IndexMap<String, LocalInfo>,
    local_types: HashMap<String, String>,  // variable name -> type name
    local_array_lens: HashMap<String, i64>,  // variable name -> array length
    closure_defs: HashMap<String, ClosureDef>,
    next_var: usize,
    loop_stack: Vec<(Block, Block)>,
    sret_ptr: Option<Value>,  // pointer to caller-allocated return space for struct returns
    current_module: String,  // e.g., "outer" for function "outer::get_inner"
}

impl<'a, 'b> FunctionTranslator<'a, 'b> {
    fn new(
        builder: FunctionBuilder<'b>,
        type_registry: &'a TypeRegistry,
        func_ids: &'a HashMap<DefId, FuncId>,
        func_names: &'a HashMap<String, FuncId>,
        module: &'a mut JITModule,
        struct_layouts: &'a HashMap<String, StructLayout>,
        enum_layouts: &'a HashMap<String, EnumLayout>,
        type_impls: &'a HashMap<String, Vec<ImplBlock>>,
        imports: &'a HashMap<String, String>,
        slice_param_funcs: &'a HashMap<String, Vec<usize>>,
        const_values: &'a HashMap<String, (i64, hir::Type)>,
        static_data: &'a HashMap<String, (DataId, bool, hir::Type)>,
        current_module: String,
    ) -> Self {
        Self {
            builder,
            type_registry,
            func_ids,
            func_names,
            module,
            struct_layouts,
            enum_layouts,
            type_impls,
            imports,
            slice_param_funcs,
            const_values,
            static_data,
            locals: IndexMap::new(),
            local_types: HashMap::new(),
            local_array_lens: HashMap::new(),
            closure_defs: HashMap::new(),
            current_module,
            next_var: 0,
            loop_stack: Vec::new(),
            sret_ptr: None,
        }
    }

    fn finalize(self) {
        self.builder.finalize();
    }

    /// Convert HIR type to Cranelift type
    fn hir_type_to_cl(&self, ty: &HirType) -> types::Type {
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
            TypeKind::Path(path) => {
                // Resolve via type registry
                if let Some(seg) = path.segments.first() {
                    let name = &seg.ident;
                    // Built-in collection types are pointer types
                    if name == "Vec" || name == "String" || name == "HashMap" {
                        return types::I64;
                    }
                    // Structs and enums are passed by pointer
                    if self.type_registry.is_struct(name) || self.struct_layouts.contains_key(name) {
                        return types::I64;
                    }
                    if self.type_registry.is_enum(name) || self.enum_layouts.contains_key(name) {
                        return types::I64;
                    }
                    // Resolve the type properly
                    let ty_id = self.type_registry.resolve_hir_type(ty);
                    return self.ty_id_to_cl(ty_id);
                }
                types::I64
            }
            TypeKind::Tuple(_) => types::I64,
            TypeKind::Array { .. } => types::I64,
            TypeKind::Slice(_) => types::I64,
            TypeKind::Fn { .. } => types::I64,
            _ => types::I64,
        }
    }

    /// Convert a resolved TyId to a Cranelift type
    fn ty_id_to_cl(&self, ty_id: TyId) -> types::Type {
        match self.type_registry.get(ty_id) {
            Some(Ty::Unit) => types::I64,
            Some(Ty::Bool) => types::I8,
            Some(Ty::Char) => types::I32,
            Some(Ty::Int(IntType::I8)) => types::I8,
            Some(Ty::Int(IntType::I16)) => types::I16,
            Some(Ty::Int(IntType::I32)) => types::I32,
            Some(Ty::Int(IntType::I64)) | Some(Ty::Int(IntType::Isize)) => types::I64,
            Some(Ty::Int(IntType::I128)) => types::I128,
            Some(Ty::Uint(UintType::U8)) => types::I8,
            Some(Ty::Uint(UintType::U16)) => types::I16,
            Some(Ty::Uint(UintType::U32)) => types::I32,
            Some(Ty::Uint(UintType::U64)) | Some(Ty::Uint(UintType::Usize)) => types::I64,
            Some(Ty::Uint(UintType::U128)) => types::I128,
            Some(Ty::Float(FloatType::F32)) => types::F32,
            Some(Ty::Float(FloatType::F64)) => types::F64,
            Some(Ty::Ref { .. }) | Some(Ty::Ptr { .. }) => types::I64,
            Some(Ty::Adt { .. }) => types::I64,
            Some(Ty::Tuple(_)) => types::I64,
            Some(Ty::Array { .. }) => types::I64,
            Some(Ty::Slice(_)) => types::I64,
            Some(Ty::Fn { .. }) => types::I64,
            Some(Ty::Str) => types::I64,
            Some(Ty::Param { .. }) => types::I64,
            Some(Ty::ImplTrait { concrete: Some(ty), .. }) => self.ty_id_to_cl(ty),
            Some(Ty::ImplTrait { .. }) => types::I64,
            Some(Ty::DynTrait { .. }) => types::I64,
            _ => types::I64,
        }
    }

    /// Infer Cranelift type from expression (for let bindings without type annotation)
    fn infer_expr_type(&self, expr: &Expr) -> types::Type {
        match &expr.kind {
            ExprKind::Lit(lit) => match lit {
                Literal::Bool(_) => types::I8,
                Literal::Char(_) => types::I32,
                Literal::Int(_, Some(IntType::I8)) => types::I8,
                Literal::Int(_, Some(IntType::I16)) => types::I16,
                Literal::Int(_, Some(IntType::I32)) => types::I32,
                Literal::Int(_, Some(IntType::I128)) => types::I128,
                Literal::Int(_, _) => types::I64, // Default to i64
                Literal::Uint(_, Some(UintType::U8)) => types::I8,
                Literal::Uint(_, Some(UintType::U16)) => types::I16,
                Literal::Uint(_, Some(UintType::U32)) => types::I32,
                Literal::Uint(_, Some(UintType::U128)) => types::I128,
                Literal::Uint(_, _) => types::I64,
                Literal::Float(_, Some(FloatType::F32)) => types::F32,
                Literal::Float(_, _) => types::F64,
                Literal::Str(_) | Literal::ByteStr(_) => types::I64,
            },
            ExprKind::Path(path) => {
                // Look up variable type
                if path.segments.len() == 1 {
                    if let Some(info) = self.locals.get(&path.segments[0].ident) {
                        return info.ty;
                    }
                }
                types::I64
            }
            ExprKind::Binary { op, lhs, .. } => {
                // Comparison ops return bool
                match op {
                    BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le |
                    BinaryOp::Gt | BinaryOp::Ge | BinaryOp::And | BinaryOp::Or => types::I8,
                    _ => self.infer_expr_type(lhs),
                }
            }
            ExprKind::Unary { expr: inner, .. } => self.infer_expr_type(inner),
            ExprKind::If { then_branch, .. } => {
                if let Some(expr) = &then_branch.expr {
                    self.infer_expr_type(expr)
                } else {
                    types::I64
                }
            }
            ExprKind::Ref { .. } | ExprKind::Deref(_) => types::I64,
            ExprKind::Call { .. } | ExprKind::MethodCall { .. } => types::I64,
            _ => types::I64,
        }
    }

    /// Check if a type is a floating point type
    fn is_float_type(&self, ty: types::Type) -> bool {
        ty == types::F32 || ty == types::F64
    }

    /// Infer the HIR type name from an expression (for method resolution)
    fn infer_type_name(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Path(path) => {
                if path.segments.len() == 1 {
                    // Variable - look up its type
                    let name = &path.segments[0].ident;
                    if let Some(ty) = self.local_types.get(name) {
                        return Some(ty.clone());
                    }
                    // Check if this is an enum variant (like None)
                    for (enum_name, enum_layout) in self.enum_layouts.iter() {
                        if enum_layout.variants.contains_key(name) {
                            return Some(enum_name.clone());
                        }
                    }
                    None
                } else if path.segments.len() == 2 {
                    // Enum::Variant - return the enum name
                    let type_name = &path.segments[0].ident;
                    if self.enum_layouts.contains_key(type_name) {
                        Some(type_name.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            ExprKind::Struct { path, .. } => {
                // Struct literal - type is the struct name (may be module-qualified)
                Some(path.segments.iter()
                    .map(|s| s.ident.as_str())
                    .collect::<Vec<_>>()
                    .join("::"))
            }
            ExprKind::Call { func, .. } => {
                // Function call - check if it's a constructor (Type::new or mod::Type::new pattern)
                if let ExprKind::Path(path) = &func.kind {
                    if path.segments.len() >= 2 {
                        // Type::method() or module::Type::method() - return all but last segment
                        let type_path: String = path.segments.iter()
                            .take(path.segments.len() - 1)
                            .map(|s| s.ident.as_str())
                            .collect::<Vec<_>>()
                            .join("::");
                        return Some(type_path);
                    }
                    if path.segments.len() == 1 {
                        // Single segment - check if it's an enum variant (Some, Ok, Err)
                        let name = &path.segments[0].ident;
                        for (enum_name, enum_layout) in self.enum_layouts.iter() {
                            if enum_layout.variants.contains_key(name) {
                                return Some(enum_name.clone());
                            }
                        }
                    }
                }
                None
            }
            ExprKind::MethodCall { receiver, method, .. } => {
                // Check if method returns a primitive (not the receiver type)
                match method.as_str() {
                    // Methods that return integers or booleans
                    "len" | "capacity" | "is_empty" | "is_some" | "is_none" => None,
                    // Methods that return the inner type (unwrap, pop, etc.)
                    "unwrap" | "pop" => None,
                    // Methods that return the receiver type (push returns (), but receiver is still valid)
                    _ => self.infer_type_name(receiver)
                }
            }
            ExprKind::Field { expr: inner, field } => {
                // Field access - get the field's type from the base struct's layout
                if let Some(base_type) = self.infer_type_name(inner) {
                    if let Some(layout) = self.struct_layouts.get(&base_type) {
                        if let Some(field_info) = layout.fields.get(field) {
                            return Some(field_info.type_name.clone());
                        }
                    }
                }
                None
            }
            ExprKind::Index { expr: base, .. } => {
                // Index access - infer element type from base
                // For Vec<T>, the element type is T (but we just return the inner type for now)
                let base_type = self.infer_type_name(base)?;
                if base_type.starts_with("Vec<") && base_type.ends_with(">") {
                    // Extract T from Vec<T>
                    Some(base_type[4..base_type.len()-1].to_string())
                } else {
                    // For other types, we can't easily determine element type
                    None
                }
            }
            _ => None,
        }
    }

    /// Resolve a method for a given type using TypeRegistry's unified method table
    fn resolve_method(&self, type_name: &str, method_name: &str) -> Option<FuncId> {
        // First try the TypeRegistry's method table (unified resolution with deref coercion)
        if let Some(method_info) = self.type_registry.resolve_method_by_name(type_name, method_name, 0) {
            // Look up FuncId using the DefId from the method table
            if let Some(&func_id) = self.func_ids.get(&method_info.def_id) {
                return Some(func_id);
            }
        }

        // Fallback: check local type_impls registry (for impl blocks processed during codegen)
        if let Some(impls) = self.type_impls.get(type_name) {
            for impl_block in impls {
                if let Some(method) = impl_block.methods.get(method_name) {
                    return Some(method.func_id);
                }
            }
        }

        // Final fallback to naming convention: Type_method
        let mangled_name = format!("{}_{}", type_name, method_name);
        self.func_names.get(&mangled_name).copied()
    }

    fn translate_block(&mut self, block: &HirBlock) -> Option<Value> {
        for stmt in &block.stmts {
            self.translate_stmt(stmt);
        }

        block.expr.as_ref().map(|e| self.translate_expr(e))
    }

    fn translate_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { pattern, ty, init } => {
                if let Some(init) = init {
                    // Check if init is a closure - store it for later
                    if let ExprKind::Closure { params, body, .. } = &init.kind {
                        if let PatternKind::Ident { name, .. } = &pattern.kind {
                            self.closure_defs.insert(name.clone(), ClosureDef {
                                params: params.clone(),
                                body: body.clone(),
                            });
                            // Store a dummy value in locals
                            let var = Variable::from_u32(self.next_var as u32);
                            self.next_var += 1;
                            self.builder.declare_var(var, types::I64);
                            let dummy = self.builder.ins().iconst(types::I64, 0);
                            self.builder.def_var(var, dummy);
                            self.locals.insert(name.clone(), LocalInfo { var, ty: types::I64 });
                            return;
                        }
                    }

                    // Infer type name from init expression for method resolution
                    let type_name = self.infer_type_name(init).or_else(|| {
                        // Also try to get type name from declared type annotation
                        ty.as_ref().and_then(|t| {
                            if let TypeKind::Path(path) = &t.kind {
                                // Join all segments for module-qualified types
                                Some(path.segments.iter()
                                    .map(|s| s.ident.as_str())
                                    .collect::<Vec<_>>()
                                    .join("::"))
                            } else {
                                None
                            }
                        })
                    });

                    // Track array length for slice parameter expansion
                    if let ExprKind::Array(elems) = &init.kind {
                        if let PatternKind::Ident { name, .. } = &pattern.kind {
                            self.local_array_lens.insert(name.clone(), elems.len() as i64);
                        }
                    }

                    let val = self.translate_expr(init);
                    // Determine type from annotation or infer from literal
                    let cl_ty = if let Some(declared_ty) = ty {
                        self.hir_type_to_cl(declared_ty)
                    } else {
                        self.infer_expr_type(init)
                    };
                    self.bind_pattern_with_type_name(pattern, val, cl_ty, type_name);
                }
            }
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
                self.translate_expr(expr);
            }
            StmtKind::Item(_) => {}
        }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, val: Value, ty: types::Type) {
        self.bind_pattern_with_type_name(pattern, val, ty, None);
    }

    fn bind_pattern_with_type_name(&mut self, pattern: &Pattern, val: Value, ty: types::Type, type_name: Option<String>) {
        if let PatternKind::Ident { name, .. } = &pattern.kind {
            let var = Variable::from_u32(self.next_var as u32);
            self.next_var += 1;
            self.builder.declare_var(var, ty);
            self.builder.def_var(var, val);
            self.locals.insert(name.clone(), LocalInfo { var, ty });
            if let Some(tn) = type_name {
                self.local_types.insert(name.clone(), tn);
            }
        }
    }

    fn translate_expr(&mut self, expr: &Expr) -> Value {
        match &expr.kind {
            ExprKind::Lit(lit) => self.translate_literal(lit),
            ExprKind::Path(path) => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0].ident;
                    if let Some(info) = self.locals.get(name) {
                        return self.builder.use_var(info.var);
                    }
                    // Check for const value
                    if let Some((value, _ty)) = self.const_values.get(name) {
                        return self.builder.ins().iconst(types::I64, *value);
                    }
                    // Check for static value - load from global data
                    if let Some((data_id, _mutable, _ty)) = self.static_data.get(name) {
                        let gv = self.module.declare_data_in_func(*data_id, self.builder.func);
                        let ptr = self.builder.ins().global_value(types::I64, gv);
                        return self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), ptr, 0);
                    }
                    // Check if this is a single-segment unit variant (like None)
                    for (enum_name, enum_layout) in self.enum_layouts.iter() {
                        if let Some(variant) = enum_layout.variants.get(name) {
                            if matches!(variant.kind, VariantKind::Unit) {
                                // Unit variant - allocate enum and store discriminant
                                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                                    enum_layout.size as u32,
                                    0,
                                ));
                                let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                                let disc = self.builder.ins().iconst(types::I64, variant.discriminant);
                                self.builder.ins().store(cranelift::prelude::MemFlags::new(), disc, ptr, 0);
                                return ptr;
                            }
                        }
                    }
                } else if path.segments.len() == 2 {
                    // Check for enum variant: MyEnum::Variant
                    let enum_name = &path.segments[0].ident;
                    let variant_name = &path.segments[1].ident;
                    if let Some(enum_layout) = self.enum_layouts.get(enum_name) {
                        if let Some(variant) = enum_layout.variants.get(variant_name) {
                            if matches!(variant.kind, VariantKind::Unit) {
                                // Unit variant - allocate enum and store discriminant
                                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                                    enum_layout.size as u32,
                                    0,
                                ));
                                let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                                let disc = self.builder.ins().iconst(types::I64, variant.discriminant);
                                self.builder.ins().store(cranelift::prelude::MemFlags::new(), disc, ptr, 0);
                                return ptr;
                            }
                        }
                    }
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs_val = self.translate_expr(lhs);
                let rhs_val = self.translate_expr(rhs);
                self.translate_binop(*op, lhs_val, rhs_val)
            }
            ExprKind::Assign { lhs, rhs } => {
                let rhs_val = self.translate_expr(rhs);
                match &lhs.kind {
                    // Handle simple path assignment
                    ExprKind::Path(path) => {
                        if path.segments.len() == 1 {
                            let name = &path.segments[0].ident;
                            if let Some(info) = self.locals.get(name) {
                                self.builder.def_var(info.var, rhs_val);
                            } else if let Some((data_id, mutable, _ty)) = self.static_data.get(name) {
                                // Assignment to static variable
                                if *mutable {
                                    let gv = self.module.declare_data_in_func(*data_id, self.builder.func);
                                    let ptr = self.builder.ins().global_value(types::I64, gv);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), rhs_val, ptr, 0);
                                }
                            }
                        }
                    }
                    // Handle pointer dereference assignment: *ptr = value
                    ExprKind::Deref(inner) => {
                        let ptr = self.translate_expr(inner);
                        self.builder.ins().store(cranelift::prelude::MemFlags::new(), rhs_val, ptr, 0);
                    }
                    // Handle indexed assignment: arr[i] = value or v[i] = value
                    ExprKind::Index { expr: base, index } => {
                        let base_type = self.infer_type_name(base);
                        let idx = self.translate_expr(index);

                        // Check if base is a Vec (BoltVec layout)
                        let is_vec = base_type.as_ref()
                            .map(|t| t == "Vec" || t.starts_with("Vec<"))
                            .unwrap_or(false);

                        if is_vec {
                            // Vec indexed assignment with bounds check
                            let vec_ptr = self.translate_expr(base);

                            // Load data pointer and length from Vec struct
                            let data_ptr = self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                vec_ptr,
                                0,
                            );
                            let len = self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                vec_ptr,
                                8,
                            );

                            // Bounds check
                            let in_bounds = self.builder.ins().icmp(IntCC::UnsignedLessThan, idx, len);
                            let ok_block = self.builder.create_block();
                            let panic_block = self.builder.create_block();

                            self.builder.ins().brif(in_bounds, ok_block, &[], panic_block, &[]);

                            // Panic block
                            self.builder.switch_to_block(panic_block);
                            self.builder.seal_block(panic_block);
                            let panic_msg = b"index out of bounds";
                            let msg_len = panic_msg.len();
                            let msg_slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                cranelift::prelude::StackSlotKind::ExplicitSlot,
                                msg_len as u32,
                                0,
                            ));
                            let msg_ptr = self.builder.ins().stack_addr(types::I64, msg_slot, 0);
                            for (i, &byte) in panic_msg.iter().enumerate() {
                                let byte_val = self.builder.ins().iconst(types::I8, byte as i64);
                                self.builder.ins().store(cranelift::prelude::MemFlags::new(), byte_val, msg_ptr, i as i32);
                            }
                            let msg_len_val = self.builder.ins().iconst(types::I64, msg_len as i64);
                            if let Some(&panic_fn) = self.func_names.get("bolt_panic") {
                                let panic_ref = self.module.declare_func_in_func(panic_fn, self.builder.func);
                                self.builder.ins().call(panic_ref, &[msg_ptr, msg_len_val]);
                            }
                            self.builder.ins().trap(cranelift::prelude::TrapCode::User(0));

                            // OK block - do the store
                            self.builder.switch_to_block(ok_block);
                            self.builder.seal_block(ok_block);

                            let elem_size = self.builder.ins().iconst(types::I64, 8);
                            let offset = self.builder.ins().imul(idx, elem_size);
                            let addr = self.builder.ins().iadd(data_ptr, offset);
                            self.builder.ins().store(cranelift::prelude::MemFlags::new(), rhs_val, addr, 0);
                        } else {
                            // Raw array indexed assignment
                            let ptr = self.translate_expr(base);
                            let eight = self.builder.ins().iconst(types::I64, 8);
                            let offset = self.builder.ins().imul(idx, eight);
                            let addr = self.builder.ins().iadd(ptr, offset);
                            self.builder.ins().store(cranelift::prelude::MemFlags::new(), rhs_val, addr, 0);
                        }
                    }
                    // Handle field assignment: s.field = value
                    ExprKind::Field { expr: base, field } => {
                        let ptr = self.translate_expr(base);
                        for layout in self.struct_layouts.values() {
                            if let Some(field_info) = layout.fields.get(field) {
                                self.builder.ins().store(
                                    cranelift::prelude::MemFlags::new(),
                                    rhs_val,
                                    ptr,
                                    field_info.offset as i32,
                                );
                                break;
                            }
                        }
                    }
                    _ => {}
                }
                rhs_val
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
            ExprKind::IfLet { pattern, expr, then_branch, else_branch } => {
                let scrutinee_val = self.translate_expr(expr);

                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let merge_block = self.builder.create_block();
                self.builder.append_block_param(merge_block, types::I64);

                // Check pattern - similar to match but only one pattern
                match &pattern.kind {
                    PatternKind::TupleStruct { path, elems } => {
                        // Handle enum tuple pattern like Some(x)
                        let (enum_name, variant_name) = if path.segments.len() == 2 {
                            (path.segments[0].ident.clone(), path.segments[1].ident.clone())
                        } else if path.segments.len() == 1 {
                            // Find variant in any enum
                            let vname = &path.segments[0].ident;
                            let mut found = None;
                            for (ename, elayout) in self.enum_layouts.iter() {
                                if elayout.variants.contains_key(vname) {
                                    found = Some((ename.clone(), vname.clone()));
                                    break;
                                }
                            }
                            if let Some((en, vn)) = found {
                                (en, vn)
                            } else {
                                self.builder.ins().jump(else_block, &[]);
                                self.builder.switch_to_block(then_block);
                                self.builder.seal_block(then_block);
                                let then_val = self.builder.ins().iconst(types::I64, 0);
                                self.builder.ins().jump(merge_block, &[then_val]);
                                self.builder.switch_to_block(else_block);
                                self.builder.seal_block(else_block);
                                let else_val = else_branch.as_ref()
                                    .map(|e| self.translate_expr(e))
                                    .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
                                self.builder.ins().jump(merge_block, &[else_val]);
                                self.builder.switch_to_block(merge_block);
                                self.builder.seal_block(merge_block);
                                return self.builder.block_params(merge_block)[0];
                            }
                        } else {
                            self.builder.ins().jump(else_block, &[]);
                            self.builder.switch_to_block(then_block);
                            self.builder.seal_block(then_block);
                            let then_val = self.builder.ins().iconst(types::I64, 0);
                            self.builder.ins().jump(merge_block, &[then_val]);
                            self.builder.switch_to_block(else_block);
                            self.builder.seal_block(else_block);
                            let else_val = else_branch.as_ref()
                                .map(|e| self.translate_expr(e))
                                .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
                            self.builder.ins().jump(merge_block, &[else_val]);
                            self.builder.switch_to_block(merge_block);
                            self.builder.seal_block(merge_block);
                            return self.builder.block_params(merge_block)[0];
                        };

                        if let Some(enum_layout) = self.enum_layouts.get(&enum_name) {
                            if let Some(variant) = enum_layout.variants.get(&variant_name) {
                                // Load and compare discriminant
                                let disc = self.builder.ins().load(
                                    types::I64,
                                    cranelift::prelude::MemFlags::new(),
                                    scrutinee_val,
                                    0,
                                );
                                let expected = self.builder.ins().iconst(types::I64, variant.discriminant);
                                let cmp = self.builder.ins().icmp(IntCC::Equal, disc, expected);

                                self.builder.ins().brif(cmp, then_block, &[], else_block, &[]);

                                self.builder.switch_to_block(then_block);
                                self.builder.seal_block(then_block);

                                // Bind tuple elements
                                for (idx, elem_pat) in elems.iter().enumerate() {
                                    if let PatternKind::Ident { name, .. } = &elem_pat.kind {
                                        let val = self.builder.ins().load(
                                            types::I64,
                                            cranelift::prelude::MemFlags::new(),
                                            scrutinee_val,
                                            (8 + idx * 8) as i32,
                                        );
                                        let var = Variable::from_u32(self.next_var as u32);
                                        self.next_var += 1;
                                        self.builder.declare_var(var, types::I64);
                                        self.builder.def_var(var, val);
                                        self.locals.insert(name.clone(), LocalInfo { var, ty: types::I64 });
                                    }
                                }

                                let then_val = self.translate_block(then_branch).unwrap_or_else(|| {
                                    self.builder.ins().iconst(types::I64, 0)
                                });
                                self.builder.ins().jump(merge_block, &[then_val]);
                            } else {
                                self.builder.ins().jump(else_block, &[]);
                                self.builder.switch_to_block(then_block);
                                self.builder.seal_block(then_block);
                                let then_val = self.builder.ins().iconst(types::I64, 0);
                                self.builder.ins().jump(merge_block, &[then_val]);
                            }
                        } else {
                            self.builder.ins().jump(else_block, &[]);
                            self.builder.switch_to_block(then_block);
                            self.builder.seal_block(then_block);
                            let then_val = self.builder.ins().iconst(types::I64, 0);
                            self.builder.ins().jump(merge_block, &[then_val]);
                        }
                    }
                    PatternKind::Path(path) => {
                        // Handle unit variant like None
                        if path.segments.len() == 2 {
                            let enum_name = &path.segments[0].ident;
                            let variant_name = &path.segments[1].ident;
                            if let Some(enum_layout) = self.enum_layouts.get(enum_name) {
                                if let Some(variant) = enum_layout.variants.get(variant_name) {
                                    let disc = self.builder.ins().load(
                                        types::I64,
                                        cranelift::prelude::MemFlags::new(),
                                        scrutinee_val,
                                        0,
                                    );
                                    let expected = self.builder.ins().iconst(types::I64, variant.discriminant);
                                    let cmp = self.builder.ins().icmp(IntCC::Equal, disc, expected);

                                    self.builder.ins().brif(cmp, then_block, &[], else_block, &[]);

                                    self.builder.switch_to_block(then_block);
                                    self.builder.seal_block(then_block);
                                    let then_val = self.translate_block(then_branch).unwrap_or_else(|| {
                                        self.builder.ins().iconst(types::I64, 0)
                                    });
                                    self.builder.ins().jump(merge_block, &[then_val]);
                                } else {
                                    self.builder.ins().jump(else_block, &[]);
                                    self.builder.switch_to_block(then_block);
                                    self.builder.seal_block(then_block);
                                    let then_val = self.builder.ins().iconst(types::I64, 0);
                                    self.builder.ins().jump(merge_block, &[then_val]);
                                }
                            } else {
                                self.builder.ins().jump(else_block, &[]);
                                self.builder.switch_to_block(then_block);
                                self.builder.seal_block(then_block);
                                let then_val = self.builder.ins().iconst(types::I64, 0);
                                self.builder.ins().jump(merge_block, &[then_val]);
                            }
                        } else {
                            self.builder.ins().jump(else_block, &[]);
                            self.builder.switch_to_block(then_block);
                            self.builder.seal_block(then_block);
                            let then_val = self.builder.ins().iconst(types::I64, 0);
                            self.builder.ins().jump(merge_block, &[then_val]);
                        }
                    }
                    _ => {
                        // Other patterns - just go to else
                        self.builder.ins().jump(else_block, &[]);
                        self.builder.switch_to_block(then_block);
                        self.builder.seal_block(then_block);
                        let then_val = self.builder.ins().iconst(types::I64, 0);
                        self.builder.ins().jump(merge_block, &[then_val]);
                    }
                }

                self.builder.switch_to_block(else_block);
                self.builder.seal_block(else_block);
                let else_val = else_branch.as_ref()
                    .map(|e| self.translate_expr(e))
                    .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
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
                // Coerce return value to i64 (function signature always uses i64)
                let val = self.coerce_to_type(val, types::I64);
                self.builder.ins().return_(&[val]);
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Call { func, args } => {
                // Handle closure call: (|x| x + 1)(5)
                if let ExprKind::Closure { params, body, .. } = &func.kind {
                    // Bind arguments to parameters
                    let arg_vals: Vec<(Value, types::Type)> = args.iter()
                        .map(|arg| (self.translate_expr(arg), self.infer_expr_type(arg)))
                        .collect();

                    for (i, (pattern, param_ty)) in params.iter().enumerate() {
                        if let PatternKind::Ident { name, .. } = &pattern.kind {
                            let var = Variable::from_u32(self.next_var as u32);
                            self.next_var += 1;
                            let ty = param_ty.as_ref()
                                .map(|t| self.hir_type_to_cl(t))
                                .unwrap_or_else(|| arg_vals.get(i).map(|(_, t)| *t).unwrap_or(types::I64));
                            self.builder.declare_var(var, ty);
                            let val = arg_vals.get(i).map(|(v, _)| *v)
                                .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
                            self.builder.def_var(var, val);
                            self.locals.insert(name.clone(), LocalInfo { var, ty });
                        }
                    }

                    // Evaluate closure body
                    return self.translate_expr(body);
                }

                // Handle call on a variable that might be a closure
                if let ExprKind::Path(path) = &func.kind {
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident;
                        // Check if this is a closure stored in locals
                        if let Some(closure) = self.closure_defs.get(name).cloned() {
                            // Bind arguments to parameters
                            let arg_vals: Vec<(Value, types::Type)> = args.iter()
                                .map(|arg| (self.translate_expr(arg), self.infer_expr_type(arg)))
                                .collect();

                            for (i, (pattern, param_ty)) in closure.params.iter().enumerate() {
                                if let PatternKind::Ident { name: param_name, .. } = &pattern.kind {
                                    let var = Variable::from_u32(self.next_var as u32);
                                    self.next_var += 1;
                                    let ty = param_ty.as_ref()
                                        .map(|t| self.hir_type_to_cl(t))
                                        .unwrap_or_else(|| arg_vals.get(i).map(|(_, t)| *t).unwrap_or(types::I64));
                                    self.builder.declare_var(var, ty);
                                    let val = arg_vals.get(i).map(|(v, _)| *v)
                                        .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
                                    self.builder.def_var(var, val);
                                    self.locals.insert(param_name.clone(), LocalInfo { var, ty });
                                }
                            }

                            // Evaluate closure body
                            return self.translate_expr(&closure.body);
                        }
                    }
                }

                // Try to resolve function name from path
                if let ExprKind::Path(path) = &func.kind {
                    // Handle multi-segment paths: mymod::func, mymod::submod::func, Type::method
                    if path.segments.len() >= 2 {
                        // Build full path string (e.g., "geometry::shapes::triangle_area")
                        let full_path: String = path.segments.iter()
                            .map(|s| s.ident.as_str())
                            .collect::<Vec<_>>()
                            .join("::");

                        // Try module-qualified function first (works for any depth)
                        let resolved_func_id = self.func_names.get(&full_path).copied()
                            // If not found, try prepending current_module for relative paths
                            .or_else(|| {
                                if !self.current_module.is_empty() {
                                    let qualified_path = format!("{}::{}", self.current_module, full_path);
                                    self.func_names.get(&qualified_path).copied()
                                } else {
                                    None
                                }
                            });

                        if let Some(func_id) = resolved_func_id {
                            let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);

                            // Translate arguments
                            let arg_vals: Vec<Value> = args.iter()
                                .map(|arg| self.translate_expr(arg))
                                .collect();

                            let call = self.builder.ins().call(func_ref, &arg_vals);
                            let results = self.builder.inst_results(call);
                            if !results.is_empty() {
                                return results[0];
                            }
                        }
                    }

                    // Handle 3+ segment paths for module::Type::method calls
                    if path.segments.len() >= 3 {
                        // Split: first n-1 segments = type, last segment = method
                        // e.g., mymod::Point::new -> type = "mymod::Point", method = "new"
                        let type_segments: Vec<_> = path.segments.iter()
                            .take(path.segments.len() - 1)
                            .map(|s| s.ident.as_str())
                            .collect();
                        let type_name = type_segments.join("::");
                        let method_name = &path.segments.last().unwrap().ident;

                        // Check for module-qualified struct static method
                        let mangled_name = format!("{}_{}", type_name, method_name);
                        if let Some(&func_id) = self.func_names.get(&mangled_name) {
                            let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);

                            // Check if this function returns a struct (needs sret)
                            let returns_struct = self.struct_layouts.contains_key(&type_name);

                            let mut arg_vals = Vec::new();

                            if returns_struct {
                                // Allocate space for return struct and pass as first arg
                                let layout = self.struct_layouts.get(&type_name).unwrap();
                                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                                    layout.size as u32,
                                    0,
                                ));
                                let sret = self.builder.ins().stack_addr(types::I64, slot, 0);
                                arg_vals.push(sret);
                            }

                            // Get function signature for type coercion
                            let sig = self.module.declarations().get_function_decl(func_id).signature.clone();
                            let param_offset = if returns_struct { 1 } else { 0 };

                            // Add regular arguments with type coercion
                            for (i, arg) in args.iter().enumerate() {
                                let val = self.translate_expr(arg);
                                let param_idx = i + param_offset;
                                if param_idx < sig.params.len() {
                                    let expected_ty = sig.params[param_idx].value_type;
                                    let coerced = self.coerce_to_type(val, expected_ty);
                                    arg_vals.push(coerced);
                                } else {
                                    arg_vals.push(val);
                                }
                            }

                            let call = self.builder.ins().call(func_ref, &arg_vals);
                            let results = self.builder.inst_results(call);
                            if returns_struct {
                                // For struct-returning functions, return the sret pointer
                                return arg_vals[0];
                            } else if !results.is_empty() {
                                return results[0];
                            }
                        }
                    }

                    // Handle 2-segment paths for enums and struct methods
                    if path.segments.len() == 2 {
                        let type_name = &path.segments[0].ident;
                        let method_name = &path.segments[1].ident;

                        // Check for enum tuple variant: MyEnum::Variant(x, y)
                        if let Some(enum_layout) = self.enum_layouts.get(type_name) {
                            if let Some(variant) = enum_layout.variants.get(method_name) {
                                if let VariantKind::Tuple(field_count) = variant.kind {
                                    // Allocate enum
                                    let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                                        enum_layout.size as u32,
                                        0,
                                    ));
                                    let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);

                                    // Store discriminant at offset 0
                                    let disc = self.builder.ins().iconst(types::I64, variant.discriminant);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), disc, ptr, 0);

                                    // Store tuple fields at offset 8, 16, 24, ...
                                    for (i, arg) in args.iter().take(field_count).enumerate() {
                                        let val = self.translate_expr(arg);
                                        self.builder.ins().store(
                                            cranelift::prelude::MemFlags::new(),
                                            val,
                                            ptr,
                                            (8 + i * 8) as i32,
                                        );
                                    }
                                    return ptr;
                                }
                            }
                        }

                        // Handle built-in Vec static methods
                        if type_name == "Vec" {
                            return match method_name.as_str() {
                                "new" => {
                                    // Vec::new() - allocate stack slot for BoltVec (24 bytes: ptr, len, cap)
                                    let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                                        24,
                                        0,
                                    ));
                                    let vec_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                                    // Initialize to zero (null ptr, 0 len, 0 cap)
                                    let zero = self.builder.ins().iconst(types::I64, 0);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, vec_ptr, 0);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, vec_ptr, 8);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, vec_ptr, 16);
                                    vec_ptr
                                }
                                "with_capacity" => {
                                    // Vec::with_capacity(n) - allocate with initial capacity
                                    let cap = args.first()
                                        .map(|arg| self.translate_expr(arg))
                                        .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
                                    let elem_size_const = 8i64;

                                    // Allocate stack slot for BoltVec struct (24 bytes)
                                    let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                                        24,
                                        0,
                                    ));
                                    let vec_struct_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);

                                    // Calculate total size: cap * elem_size
                                    let elem_size = self.builder.ins().iconst(types::I64, elem_size_const);
                                    let total_size = self.builder.ins().imul(cap, elem_size);

                                    // Allocate heap memory for data
                                    let alloc_fn = self.func_names.get("bolt_malloc")
                                        .expect("bolt_malloc not registered");
                                    let alloc_ref = self.module.declare_func_in_func(*alloc_fn, self.builder.func);
                                    let alloc_call = self.builder.ins().call(alloc_ref, &[total_size]);
                                    let data_ptr = self.builder.inst_results(alloc_call)[0];

                                    // Store ptr, len=0, cap in the Vec struct
                                    let zero = self.builder.ins().iconst(types::I64, 0);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), data_ptr, vec_struct_ptr, 0);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, vec_struct_ptr, 8);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), cap, vec_struct_ptr, 16);
                                    vec_struct_ptr
                                }
                                _ => self.builder.ins().iconst(types::I64, 0)
                            };
                        }

                        // Handle built-in String static methods
                        if type_name == "String" {
                            return match method_name.as_str() {
                                "new" => {
                                    // String::new() - same as Vec::new() for bytes
                                    let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                                        24,
                                        0,
                                    ));
                                    let str_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                                    let zero = self.builder.ins().iconst(types::I64, 0);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, str_ptr, 0);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, str_ptr, 8);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, str_ptr, 16);
                                    str_ptr
                                }
                                "from" => {
                                    // String::from("literal") - create from string literal
                                    if let Some(arg) = args.first() {
                                        if let ExprKind::Lit(Literal::Str(s)) = &arg.kind {
                                            let bytes = s.as_bytes();
                                            let len = bytes.len();
                                            let len_val = self.builder.ins().iconst(types::I64, len as i64);

                                            // Allocate string struct slot (24 bytes: ptr, len, cap)
                                            let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                                cranelift::prelude::StackSlotKind::ExplicitSlot,
                                                24,
                                                0,
                                            ));
                                            let str_struct_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);

                                            if len == 0 {
                                                // Empty string - just zero out the struct
                                                let zero = self.builder.ins().iconst(types::I64, 0);
                                                self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, str_struct_ptr, 0);
                                                self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, str_struct_ptr, 8);
                                                self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, str_struct_ptr, 16);
                                            } else {
                                                // Allocate heap memory for string data
                                                let alloc_fn = self.func_names.get("bolt_malloc")
                                                    .expect("bolt_malloc not registered");
                                                let alloc_ref = self.module.declare_func_in_func(*alloc_fn, self.builder.func);
                                                let alloc_call = self.builder.ins().call(alloc_ref, &[len_val]);
                                                let heap_ptr = self.builder.inst_results(alloc_call)[0];

                                                // Copy string literal bytes to heap (inline)
                                                for i in 0..len {
                                                    let byte_val = self.builder.ins().iconst(types::I8, bytes[i] as i64);
                                                    self.builder.ins().store(
                                                        cranelift::prelude::MemFlags::new(),
                                                        byte_val,
                                                        heap_ptr,
                                                        i as i32,
                                                    );
                                                }

                                                // Store ptr, len, cap in the String struct
                                                self.builder.ins().store(cranelift::prelude::MemFlags::new(), heap_ptr, str_struct_ptr, 0);
                                                self.builder.ins().store(cranelift::prelude::MemFlags::new(), len_val, str_struct_ptr, 8);
                                                self.builder.ins().store(cranelift::prelude::MemFlags::new(), len_val, str_struct_ptr, 16);
                                            }
                                            return str_struct_ptr;
                                        }
                                    }
                                    self.builder.ins().iconst(types::I64, 0)
                                }
                                _ => self.builder.ins().iconst(types::I64, 0)
                            };
                        }

                        // Handle built-in HashMap static methods
                        if type_name == "HashMap" {
                            return match method_name.as_str() {
                                "new" => {
                                    // HashMap::new() - allocate stack slot for BoltHashMap (24 bytes: entries, len, cap)
                                    let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                                        24,
                                        0,
                                    ));
                                    let map_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                                    // Initialize to zero (null entries, 0 len, 0 cap)
                                    let zero = self.builder.ins().iconst(types::I64, 0);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, map_ptr, 0);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, map_ptr, 8);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), zero, map_ptr, 16);
                                    map_ptr
                                }
                                "with_capacity" => {
                                    // HashMap::with_capacity(n) - allocate with initial capacity
                                    let cap = args.first()
                                        .map(|arg| self.translate_expr(arg))
                                        .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));

                                    // Call runtime function
                                    let func_id = self.func_names.get("bolt_hashmap_with_capacity")
                                        .expect("bolt_hashmap_with_capacity not registered");
                                    let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);

                                    // Allocate stack slot for result
                                    let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                                        24,
                                        0,
                                    ));
                                    let map_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);

                                    self.builder.ins().call(func_ref, &[map_ptr, cap]);
                                    map_ptr
                                }
                                _ => self.builder.ins().iconst(types::I64, 0)
                            };
                        }

                        // Check for struct static method: Point::new(x, y)
                        let mangled_name = format!("{}_{}", type_name, method_name);
                        if let Some(&func_id) = self.func_names.get(&mangled_name) {
                            let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);

                            // Check if this function returns a struct (needs sret)
                            let returns_struct = self.struct_layouts.contains_key(type_name);

                            let mut arg_vals = Vec::new();

                            if returns_struct {
                                // Allocate space for return struct and pass as first arg
                                let layout = self.struct_layouts.get(type_name).unwrap();
                                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                                    layout.size as u32,
                                    0,
                                ));
                                let sret = self.builder.ins().stack_addr(types::I64, slot, 0);
                                arg_vals.push(sret);
                            }

                            // Get function signature for type coercion
                            let sig = self.module.declarations().get_function_decl(func_id).signature.clone();
                            let param_offset = if returns_struct { 1 } else { 0 };

                            // Add regular arguments with type coercion
                            for (i, arg) in args.iter().enumerate() {
                                let val = self.translate_expr(arg);
                                let param_idx = i + param_offset;
                                if param_idx < sig.params.len() {
                                    let expected_ty = sig.params[param_idx].value_type;
                                    let coerced = self.coerce_to_type(val, expected_ty);
                                    arg_vals.push(coerced);
                                } else {
                                    arg_vals.push(val);
                                }
                            }

                            let call = self.builder.ins().call(func_ref, &arg_vals);
                            let results = self.builder.inst_results(call);
                            if returns_struct {
                                // For struct-returning functions, return the sret pointer
                                return arg_vals[0];
                            } else if !results.is_empty() {
                                return results[0];
                            }
                        }
                    }
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident;

                        // Check if this is an enum variant constructor (e.g., Some(42))
                        for (enum_name, enum_layout) in self.enum_layouts.iter() {
                            if let Some(variant) = enum_layout.variants.get(name) {
                                if let VariantKind::Tuple(field_count) = variant.kind {
                                    // Allocate enum
                                    let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                                        enum_layout.size as u32,
                                        0,
                                    ));
                                    let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);

                                    // Store discriminant at offset 0
                                    let disc = self.builder.ins().iconst(types::I64, variant.discriminant);
                                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), disc, ptr, 0);

                                    // Store tuple fields at offset 8, 16, 24, ...
                                    for (i, arg) in args.iter().take(field_count).enumerate() {
                                        let val = self.translate_expr(arg);
                                        let val64 = self.coerce_to_type(val, types::I64);
                                        self.builder.ins().store(
                                            cranelift::prelude::MemFlags::new(),
                                            val64,
                                            ptr,
                                            (8 + i * 8) as i32,
                                        );
                                    }

                                    return ptr;
                                }
                            }
                        }

                        // First check if this name is an import alias
                        let resolved_name = self.imports.get(name).cloned().unwrap_or_else(|| name.clone());

                        // Handle built-in print function
                        if resolved_name == "print" && args.len() == 1 {
                            let arg_type_name = self.infer_type_name(&args[0]);
                            let arg_val = self.translate_expr(&args[0]);
                            let arg_ty = self.infer_expr_type(&args[0]);

                            // Check if printing a String
                            if arg_type_name.as_ref().map(|t| t == "String").unwrap_or(false) {
                                // Load ptr and len from String struct
                                let str_ptr = self.builder.ins().load(
                                    types::I64,
                                    cranelift::prelude::MemFlags::new(),
                                    arg_val,
                                    0,
                                );
                                let str_len = self.builder.ins().load(
                                    types::I64,
                                    cranelift::prelude::MemFlags::new(),
                                    arg_val,
                                    8,
                                );
                                if let Some(&func_id) = self.func_names.get("bolt_print_str") {
                                    let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
                                    self.builder.ins().call(func_ref, &[str_ptr, str_len]);
                                }
                                // Print newline
                                if let Some(&func_id) = self.func_names.get("bolt_print_newline") {
                                    let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
                                    self.builder.ins().call(func_ref, &[]);
                                }
                                return self.builder.ins().iconst(types::I64, 0);
                            }

                            let fn_name = if arg_ty == types::F64 || arg_ty == types::F32 {
                                "bolt_print_f64"
                            } else {
                                "bolt_print_i64"
                            };

                            if let Some(&func_id) = self.func_names.get(fn_name) {
                                let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
                                self.builder.ins().call(func_ref, &[arg_val]);
                            }
                            return self.builder.ins().iconst(types::I64, 0);
                        }

                        if let Some(&func_id) = self.func_names.get(&resolved_name) {
                            // Get function reference
                            let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);

                            // Check if function name follows Type_method pattern and returns struct
                            let returns_struct = if let Some(underscore_pos) = resolved_name.rfind('_') {
                                let type_name = &resolved_name[..underscore_pos];
                                self.struct_layouts.contains_key(type_name)
                            } else {
                                false
                            };

                            let mut arg_vals = Vec::new();

                            if returns_struct {
                                // Extract type name and allocate sret space
                                let underscore_pos = resolved_name.rfind('_').unwrap();
                                let type_name = &resolved_name[..underscore_pos];
                                let layout = self.struct_layouts.get(type_name).unwrap();
                                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                                    layout.size as u32,
                                    0,
                                ));
                                let sret = self.builder.ins().stack_addr(types::I64, slot, 0);
                                arg_vals.push(sret);
                            }

                            // Get function signature for type coercion
                            let sig = self.module.declarations().get_function_decl(func_id).signature.clone();
                            let param_offset = if returns_struct { 1 } else { 0 };

                            // Check if this function has slice parameters
                            let slice_params = self.slice_param_funcs.get(&resolved_name).cloned().unwrap_or_default();

                            // Add regular arguments with type coercion and slice expansion
                            let mut sig_param_idx = param_offset;
                            for (i, arg) in args.iter().enumerate() {
                                if slice_params.contains(&i) {
                                    // This argument should be expanded to (ptr, len) fat pointer
                                    let arr_ptr = self.translate_expr(arg);

                                    // Get the array length from the expression
                                    let arr_len = if let ExprKind::Ref { expr: inner, .. } = &arg.kind {
                                        // Check if inner is an array literal
                                        if let ExprKind::Array(elems) = &inner.kind {
                                            elems.len() as i64
                                        } else if let ExprKind::Path(path) = &inner.kind {
                                            // Try to find the array length from local_types
                                            let name = path.segments.last()
                                                .map(|s| s.ident.clone())
                                                .unwrap_or_default();
                                            self.local_array_lens.get(&name).copied().unwrap_or(0)
                                        } else {
                                            0
                                        }
                                    } else {
                                        0
                                    };

                                    arg_vals.push(arr_ptr);
                                    arg_vals.push(self.builder.ins().iconst(types::I64, arr_len));
                                    sig_param_idx += 2;
                                } else {
                                    let val = self.translate_expr(arg);
                                    if sig_param_idx < sig.params.len() {
                                        let expected_ty = sig.params[sig_param_idx].value_type;
                                        let coerced = self.coerce_to_type(val, expected_ty);
                                        arg_vals.push(coerced);
                                    } else {
                                        arg_vals.push(val);
                                    }
                                    sig_param_idx += 1;
                                }
                            }

                            // Emit call
                            let call = self.builder.ins().call(func_ref, &arg_vals);
                            let results = self.builder.inst_results(call);
                            if returns_struct {
                                // Return the sret pointer
                                return arg_vals[0];
                            } else if !results.is_empty() {
                                return results[0];
                            }
                        }
                    }
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::MethodCall { receiver, method, args } => {
                // First, try to infer the receiver type for proper method resolution
                let type_name = self.infer_type_name(receiver);
                let recv_val = self.translate_expr(receiver);

                // Handle built-in Vec methods
                if let Some(ref tn) = type_name {
                    if tn == "Vec" || tn.starts_with("Vec<") {
                        return self.translate_vec_method(recv_val, method, args);
                    }
                    if tn == "String" {
                        return self.translate_string_method(recv_val, method, args);
                    }
                    if tn == "HashMap" || tn.starts_with("HashMap<") {
                        return self.translate_hashmap_method(recv_val, method, args);
                    }
                    if tn == "Slice" || tn.starts_with("&[") {
                        return self.translate_slice_method(receiver, method, args);
                    }
                }

                // Check if receiver is a slice local (has {name}_len companion)
                if let ExprKind::Path(path) = &receiver.kind {
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident;
                        let len_name = format!("{}_len", name);
                        if self.locals.contains_key(&len_name) {
                            return self.translate_slice_method(receiver, method, args);
                        }
                    }
                }

                // Check if receiver is a local array (for .len() method)
                if method == "len" {
                    if let ExprKind::Path(path) = &receiver.kind {
                        if path.segments.len() == 1 {
                            let name = &path.segments[0].ident;
                            if let Some(&arr_len) = self.local_array_lens.get(name) {
                                return self.builder.ins().iconst(types::I64, arr_len);
                            }
                        }
                    }
                }

                // Handle Option methods (unwrap, is_some, is_none, etc.)
                // Check if receiver type is Option
                let is_option = type_name.as_ref().map(|tn| tn == "Option").unwrap_or(false)
                    && self.enum_layouts.get("Option").is_some();
                if is_option {
                    match method.as_str() {
                        "unwrap" => {
                            // Load discriminant (at offset 0)
                            let disc = self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                0,
                            );
                            // Check if it's Some (discriminant 0 for Some in our layout)
                            let some_disc = self.enum_layouts.get("Option")
                                .and_then(|l| l.variants.get("Some"))
                                .map(|v| v.discriminant)
                                .unwrap_or(0);
                            let expected = self.builder.ins().iconst(types::I64, some_disc);
                            let is_some = self.builder.ins().icmp(IntCC::Equal, disc, expected);

                            let some_block = self.builder.create_block();
                            let panic_block = self.builder.create_block();

                            self.builder.ins().brif(is_some, some_block, &[], panic_block, &[]);

                            // Panic block - call abort
                            self.builder.switch_to_block(panic_block);
                            self.builder.seal_block(panic_block);
                            if let Some(&abort_fn) = self.func_names.get("bolt_abort") {
                                let abort_ref = self.module.declare_func_in_func(abort_fn, self.builder.func);
                                self.builder.ins().call(abort_ref, &[]);
                            }
                            self.builder.ins().trap(cranelift::prelude::TrapCode::User(0));

                            // Some block - extract and return payload
                            self.builder.switch_to_block(some_block);
                            self.builder.seal_block(some_block);
                            let payload = self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                8,  // payload at offset 8
                            );
                            return payload;
                        }
                        "is_some" => {
                            let disc = self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                0,
                            );
                            let some_disc = self.enum_layouts.get("Option")
                                .and_then(|l| l.variants.get("Some"))
                                .map(|v| v.discriminant)
                                .unwrap_or(0);
                            let expected = self.builder.ins().iconst(types::I64, some_disc);
                            return self.builder.ins().icmp(IntCC::Equal, disc, expected);
                        }
                        "is_none" => {
                            let disc = self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                0,
                            );
                            let none_disc = self.enum_layouts.get("Option")
                                .and_then(|l| l.variants.get("None"))
                                .map(|v| v.discriminant)
                                .unwrap_or(1);
                            let expected = self.builder.ins().iconst(types::I64, none_disc);
                            return self.builder.ins().icmp(IntCC::Equal, disc, expected);
                        }
                        _ => {}
                    }
                }

                // Handle Vec methods (push, pop, len, is_empty, etc.)
                let is_vec = type_name.as_ref().map(|tn| tn.starts_with("Vec")).unwrap_or(false);
                if is_vec {
                    match method.as_str() {
                        "len" => {
                            // Vec layout: ptr(0), len(8), cap(16)
                            return self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                8,  // len is at offset 8
                            );
                        }
                        "is_empty" => {
                            let len = self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                8,
                            );
                            let zero = self.builder.ins().iconst(types::I64, 0);
                            return self.builder.ins().icmp(IntCC::Equal, len, zero);
                        }
                        "push" => {
                            // Push element: bolt_vec_push(vec_ptr, elem_ptr, elem_size=8)
                            if let Some(arg) = args.first() {
                                let arg_val = self.translate_expr(arg);
                                // Store the value to a temp slot to get a pointer
                                let slot = self.builder.create_sized_stack_slot(
                                    cranelift::prelude::StackSlotData::new(
                                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                                        8,
                                        0,
                                    ),
                                );
                                let elem_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                                self.builder.ins().store(
                                    cranelift::prelude::MemFlags::new(),
                                    arg_val,
                                    elem_ptr,
                                    0,
                                );
                                // Call bolt_vec_push
                                if let Some(&func_id) = self.func_names.get("bolt_vec_push") {
                                    let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
                                    let elem_size = self.builder.ins().iconst(types::I64, 8);
                                    self.builder.ins().call(func_ref, &[recv_val, elem_ptr, elem_size]);
                                }
                            }
                            return self.builder.ins().iconst(types::I64, 0);
                        }
                        "pop" => {
                            // Pop element: bolt_vec_pop(vec_ptr, elem_size=8) -> ptr
                            if let Some(&func_id) = self.func_names.get("bolt_vec_pop") {
                                let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
                                let elem_size = self.builder.ins().iconst(types::I64, 8);
                                let call = self.builder.ins().call(func_ref, &[recv_val, elem_size]);
                                let results = self.builder.inst_results(call);
                                if !results.is_empty() {
                                    let elem_ptr = results[0];
                                    // Load the value from the pointer (if not null)
                                    // For simplicity, just load it
                                    return self.builder.ins().load(
                                        types::I64,
                                        cranelift::prelude::MemFlags::new(),
                                        elem_ptr,
                                        0,
                                    );
                                }
                            }
                            return self.builder.ins().iconst(types::I64, 0);
                        }
                        "capacity" => {
                            // cap is at offset 16
                            return self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                16,
                            );
                        }
                        "clear" => {
                            // Just set len to 0
                            let zero = self.builder.ins().iconst(types::I64, 0);
                            self.builder.ins().store(
                                cranelift::prelude::MemFlags::new(),
                                zero,
                                recv_val,
                                8,
                            );
                            return zero;
                        }
                        _ => {}
                    }
                }

                // Handle String methods (len, is_empty, push_str, etc.)
                // Note: String methods are now primarily handled by translate_string_method
                // This is a fallback for any methods not caught by the early return
                let is_string = type_name.as_ref().map(|tn| tn == "String").unwrap_or(false);
                if is_string {
                    match method.as_str() {
                        "len" => {
                            // String layout same as Vec: ptr(0), len(8), cap(16)
                            return self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                8,
                            );
                        }
                        "is_empty" => {
                            let len = self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                8,
                            );
                            let zero = self.builder.ins().iconst(types::I64, 0);
                            return self.builder.ins().icmp(IntCC::Equal, len, zero);
                        }
                        "as_ptr" => {
                            // Return the data pointer
                            return self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                recv_val,
                                0,
                            );
                        }
                        "push_str" => {
                            // s.push_str("literal") - append string
                            if let Some(arg) = args.first() {
                                if let ExprKind::Lit(Literal::Str(lit)) = &arg.kind {
                                    let bytes = lit.as_bytes();
                                    let len = bytes.len();

                                    // Store literal to stack for pointer
                                    let lit_slot = self.builder.create_sized_stack_slot(
                                        cranelift::prelude::StackSlotData::new(
                                            cranelift::prelude::StackSlotKind::ExplicitSlot,
                                            len as u32,
                                            0,
                                        ),
                                    );
                                    let lit_ptr = self.builder.ins().stack_addr(types::I64, lit_slot, 0);
                                    for (i, &byte) in bytes.iter().enumerate() {
                                        let byte_val = self.builder.ins().iconst(types::I8, byte as i64);
                                        self.builder.ins().store(
                                            cranelift::prelude::MemFlags::new(),
                                            byte_val,
                                            lit_ptr,
                                            i as i32,
                                        );
                                    }

                                    // Call bolt_string_push_str
                                    if let Some(&func_id) = self.func_names.get("bolt_string_push_str") {
                                        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
                                        let len_val = self.builder.ins().iconst(types::I64, len as i64);
                                        self.builder.ins().call(func_ref, &[recv_val, lit_ptr, len_val]);
                                    }
                                }
                            }
                            return self.builder.ins().iconst(types::I64, 0);
                        }
                        "clear" => {
                            // Just set len to 0
                            let zero = self.builder.ins().iconst(types::I64, 0);
                            self.builder.ins().store(
                                cranelift::prelude::MemFlags::new(),
                                zero,
                                recv_val,
                                8,
                            );
                            return zero;
                        }
                        _ => {}
                    }
                }

                // Try to resolve the method using the type registry
                if let Some(ref tn) = type_name {
                    if let Some(func_id) = self.resolve_method(tn, method) {
                        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
                        let sig = self.module.declarations().get_function_decl(func_id).signature.clone();

                        // Check if this method returns a struct (sret convention)
                        // If sig.params.len() > args.len() + 1 (self), first param is sret
                        let expected_regular_params = args.len() + 1; // self + args
                        let returns_struct = sig.params.len() > expected_regular_params;

                        let mut arg_vals = Vec::new();

                        if returns_struct {
                            // Allocate space for struct return
                            let struct_size = self.struct_layouts.get(tn)
                                .map(|l| l.size)
                                .unwrap_or(16) as u32;
                            let slot = self.builder.create_sized_stack_slot(
                                cranelift::prelude::StackSlotData::new(
                                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                                    struct_size,
                                    0,
                                ),
                            );
                            let sret_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                            arg_vals.push(sret_ptr);
                        }

                        // Add self
                        arg_vals.push(recv_val);

                        // Add rest of args with type coercion
                        let param_offset = if returns_struct { 2 } else { 1 }; // sret + self or just self
                        for (i, arg) in args.iter().enumerate() {
                            let val = self.translate_expr(arg);
                            let param_idx = i + param_offset;
                            if param_idx < sig.params.len() {
                                let expected_ty = sig.params[param_idx].value_type;
                                let coerced = self.coerce_to_type(val, expected_ty);
                                arg_vals.push(coerced);
                            } else {
                                arg_vals.push(val);
                            }
                        }

                        let call = self.builder.ins().call(func_ref, &arg_vals);
                        if returns_struct {
                            // Return the sret pointer
                            return arg_vals[0];
                        }
                        let results = self.builder.inst_results(call);
                        if !results.is_empty() {
                            return results[0];
                        }
                    }
                }

                // Fallback: scan for any Type_method pattern that matches
                for (name, &func_id) in self.func_names.iter() {
                    if name.ends_with(&format!("_{}", method)) {
                        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
                        let sig = self.module.declarations().get_function_decl(func_id).signature.clone();

                        // Check if this method returns a struct (sret convention)
                        let expected_regular_params = args.len() + 1; // self + args
                        let returns_struct = sig.params.len() > expected_regular_params;

                        let mut arg_vals = Vec::new();

                        // Get struct size from the function name (Type_method -> Type)
                        let struct_name = name.strip_suffix(&format!("_{}", method)).unwrap_or("");

                        if returns_struct {
                            let struct_size = self.struct_layouts.get(struct_name)
                                .map(|l| l.size)
                                .unwrap_or(16) as u32;
                            let slot = self.builder.create_sized_stack_slot(
                                cranelift::prelude::StackSlotData::new(
                                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                                    struct_size,
                                    0,
                                ),
                            );
                            let sret_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                            arg_vals.push(sret_ptr);
                        }

                        // Add self
                        arg_vals.push(recv_val);

                        // Add rest of args with type coercion
                        let param_offset = if returns_struct { 2 } else { 1 };
                        for (i, arg) in args.iter().enumerate() {
                            let val = self.translate_expr(arg);
                            let param_idx = i + param_offset;
                            if param_idx < sig.params.len() {
                                let expected_ty = sig.params[param_idx].value_type;
                                let coerced = self.coerce_to_type(val, expected_ty);
                                arg_vals.push(coerced);
                            } else {
                                arg_vals.push(val);
                            }
                        }

                        let call = self.builder.ins().call(func_ref, &arg_vals);
                        if returns_struct {
                            return arg_vals[0];
                        }
                        let results = self.builder.inst_results(call);
                        if !results.is_empty() {
                            return results[0];
                        }
                    }
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Tuple(elems) => {
                if elems.is_empty() {
                    self.builder.ins().iconst(types::I64, 0)
                } else {
                    // Allocate stack space for tuple (8 bytes per element)
                    let size = elems.len() * 8;
                    let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                        size as u32,
                        0,
                    ));
                    let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);

                    // Store each element at its offset
                    for (i, elem) in elems.iter().enumerate() {
                        let val = self.translate_expr(elem);
                        self.builder.ins().store(
                            cranelift::prelude::MemFlags::new(),
                            val,
                            ptr,
                            (i * 8) as i32,
                        );
                    }
                    ptr
                }
            }
            ExprKind::Loop { body, .. } => {
                let header_block = self.builder.create_block();
                let exit_block = self.builder.create_block();
                self.builder.append_block_param(exit_block, types::I64);
                
                self.builder.ins().jump(header_block, &[]);
                
                self.builder.switch_to_block(header_block);
                // Don't seal header yet
                
                self.loop_stack.push((header_block, exit_block));
                self.translate_block(body);
                self.loop_stack.pop();
                
                self.builder.ins().jump(header_block, &[]);
                
                // Now seal header after back edge
                self.builder.seal_block(header_block);
                
                self.builder.switch_to_block(exit_block);
                self.builder.seal_block(exit_block);
                
                self.builder.block_params(exit_block)[0]
            }
            ExprKind::While { cond, body, .. } => {
                let header_block = self.builder.create_block();
                let body_block = self.builder.create_block();
                let exit_block = self.builder.create_block();
                
                self.builder.ins().jump(header_block, &[]);
                
                self.builder.switch_to_block(header_block);
                // Don't seal header yet - back edge not added
                
                let cond_val = self.translate_expr(cond);
                self.builder.ins().brif(cond_val, body_block, &[], exit_block, &[]);
                
                self.builder.switch_to_block(body_block);
                self.builder.seal_block(body_block);
                
                self.loop_stack.push((header_block, exit_block));
                self.translate_block(body);
                self.loop_stack.pop();
                
                self.builder.ins().jump(header_block, &[]);
                
                // Now seal header after back edge
                self.builder.seal_block(header_block);
                
                self.builder.switch_to_block(exit_block);
                self.builder.seal_block(exit_block);

                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::WhileLet { pattern, expr, body, .. } => {
                let header_block = self.builder.create_block();
                let body_block = self.builder.create_block();
                let exit_block = self.builder.create_block();

                self.builder.ins().jump(header_block, &[]);

                self.builder.switch_to_block(header_block);
                // Don't seal header yet - back edge not added

                let scrutinee_val = self.translate_expr(expr);

                // Check pattern - dispatch to body or exit
                match &pattern.kind {
                    PatternKind::TupleStruct { path, elems } => {
                        let (enum_name, variant_name) = if path.segments.len() == 2 {
                            (path.segments[0].ident.clone(), path.segments[1].ident.clone())
                        } else if path.segments.len() == 1 {
                            let vname = &path.segments[0].ident;
                            let mut found = None;
                            for (ename, elayout) in self.enum_layouts.iter() {
                                if elayout.variants.contains_key(vname) {
                                    found = Some((ename.clone(), vname.clone()));
                                    break;
                                }
                            }
                            if let Some((en, vn)) = found {
                                (en, vn)
                            } else {
                                self.builder.ins().jump(exit_block, &[]);
                                self.builder.switch_to_block(body_block);
                                self.builder.seal_block(body_block);
                                self.builder.ins().jump(header_block, &[]);
                                self.builder.seal_block(header_block);
                                self.builder.switch_to_block(exit_block);
                                self.builder.seal_block(exit_block);
                                return self.builder.ins().iconst(types::I64, 0);
                            }
                        } else {
                            self.builder.ins().jump(exit_block, &[]);
                            self.builder.switch_to_block(body_block);
                            self.builder.seal_block(body_block);
                            self.builder.ins().jump(header_block, &[]);
                            self.builder.seal_block(header_block);
                            self.builder.switch_to_block(exit_block);
                            self.builder.seal_block(exit_block);
                            return self.builder.ins().iconst(types::I64, 0);
                        };

                        if let Some(enum_layout) = self.enum_layouts.get(&enum_name) {
                            if let Some(variant) = enum_layout.variants.get(&variant_name) {
                                let disc = self.builder.ins().load(
                                    types::I64,
                                    cranelift::prelude::MemFlags::new(),
                                    scrutinee_val,
                                    0,
                                );
                                let expected = self.builder.ins().iconst(types::I64, variant.discriminant);
                                let cmp = self.builder.ins().icmp(IntCC::Equal, disc, expected);

                                self.builder.ins().brif(cmp, body_block, &[], exit_block, &[]);

                                self.builder.switch_to_block(body_block);
                                self.builder.seal_block(body_block);

                                // Bind tuple elements
                                for (idx, elem_pat) in elems.iter().enumerate() {
                                    if let PatternKind::Ident { name, .. } = &elem_pat.kind {
                                        let val = self.builder.ins().load(
                                            types::I64,
                                            cranelift::prelude::MemFlags::new(),
                                            scrutinee_val,
                                            (8 + idx * 8) as i32,
                                        );
                                        let var = Variable::from_u32(self.next_var as u32);
                                        self.next_var += 1;
                                        self.builder.declare_var(var, types::I64);
                                        self.builder.def_var(var, val);
                                        self.locals.insert(name.clone(), LocalInfo { var, ty: types::I64 });
                                    }
                                }

                                self.loop_stack.push((header_block, exit_block));
                                self.translate_block(body);
                                self.loop_stack.pop();

                                self.builder.ins().jump(header_block, &[]);
                            } else {
                                self.builder.ins().jump(exit_block, &[]);
                                self.builder.switch_to_block(body_block);
                                self.builder.seal_block(body_block);
                                self.builder.ins().jump(header_block, &[]);
                            }
                        } else {
                            self.builder.ins().jump(exit_block, &[]);
                            self.builder.switch_to_block(body_block);
                            self.builder.seal_block(body_block);
                            self.builder.ins().jump(header_block, &[]);
                        }
                    }
                    _ => {
                        self.builder.ins().jump(exit_block, &[]);
                        self.builder.switch_to_block(body_block);
                        self.builder.seal_block(body_block);
                        self.builder.ins().jump(header_block, &[]);
                    }
                }

                self.builder.seal_block(header_block);

                self.builder.switch_to_block(exit_block);
                self.builder.seal_block(exit_block);

                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Break { value, .. } => {
                let val = value.as_ref()
                    .map(|v| self.translate_expr(v))
                    .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
                if let Some(&(_, exit_block)) = self.loop_stack.last() {
                    self.builder.ins().jump(exit_block, &[val]);
                    // Create unreachable block for any code after break
                    let unreachable = self.builder.create_block();
                    self.builder.switch_to_block(unreachable);
                    self.builder.seal_block(unreachable);
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Continue { .. } => {
                if let Some(&(header_block, _)) = self.loop_stack.last() {
                    self.builder.ins().jump(header_block, &[]);
                    // Create unreachable block for any code after continue
                    let unreachable = self.builder.create_block();
                    self.builder.switch_to_block(unreachable);
                    self.builder.seal_block(unreachable);
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Struct { path, fields, .. } => {
                // Get struct name from path (may be module-qualified like mymod::Point)
                let struct_name: String = path.segments.iter()
                    .map(|s| s.ident.as_str())
                    .collect::<Vec<_>>()
                    .join("::");
                if let Some(layout) = self.struct_layouts.get(&struct_name) {
                    // Use sret pointer if available (for struct return functions)
                    // Otherwise allocate a local stack slot
                    let ptr = if let Some(sret) = self.sret_ptr {
                        // Clear sret_ptr so nested struct expressions don't use it
                        self.sret_ptr = None;
                        sret
                    } else {
                        let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                            cranelift::prelude::StackSlotKind::ExplicitSlot,
                            layout.size as u32,
                            0,
                        ));
                        self.builder.ins().stack_addr(types::I64, slot, 0)
                    };

                    // Store each field
                    for (field_name, field_expr) in fields {
                        if let Some(field_info) = layout.fields.get(field_name) {
                            // Check if this field is a struct type
                            let is_struct_field = self.struct_layouts.contains_key(&field_info.type_name);
                            if is_struct_field {
                                // For struct fields, we need to copy the entire struct
                                // not just store a pointer
                                let src_ptr = self.translate_expr(field_expr);
                                if let Some(nested_layout) = self.struct_layouts.get(&field_info.type_name) {
                                    // Copy each 8-byte chunk
                                    for i in 0..(nested_layout.size / 8) {
                                        let src_offset = (i * 8) as i32;
                                        let val = self.builder.ins().load(
                                            types::I64,
                                            cranelift::prelude::MemFlags::new(),
                                            src_ptr,
                                            src_offset,
                                        );
                                        self.builder.ins().store(
                                            cranelift::prelude::MemFlags::new(),
                                            val,
                                            ptr,
                                            field_info.offset as i32 + src_offset,
                                        );
                                    }
                                }
                            } else {
                                // For scalar fields, extend to i64 and store
                                // (struct layout uses 8-byte slots for all fields)
                                let val = self.translate_expr(field_expr);
                                let val64 = self.coerce_to_type(val, types::I64);
                                self.builder.ins().store(
                                    cranelift::prelude::MemFlags::new(),
                                    val64,
                                    ptr,
                                    field_info.offset as i32,
                                );
                            }
                        }
                    }

                    return ptr;
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Field { expr: base, field } => {
                let ptr = self.translate_expr(base);

                // Check for tuple field access (numeric index like .0, .1, .2)
                if let Ok(index) = field.parse::<usize>() {
                    // Tuple field access - load from offset index * 8
                    return self.builder.ins().load(
                        types::I64,
                        cranelift::prelude::MemFlags::new(),
                        ptr,
                        (index * 8) as i32,
                    );
                }

                // First try to infer the base type and look up in that specific layout
                if let Some(base_type) = self.infer_type_name(base) {
                    if let Some(layout) = self.struct_layouts.get(&base_type) {
                        if let Some(field_info) = layout.fields.get(field) {
                            // Check if field type is a struct AT ACCESS TIME (not registration time)
                            // This handles HashMap iteration order issues
                            let is_struct_field = self.struct_layouts.contains_key(&field_info.type_name);
                            if is_struct_field {
                                // For struct-typed fields, return pointer to the nested struct
                                let offset_val = self.builder.ins().iconst(types::I64, field_info.offset as i64);
                                return self.builder.ins().iadd(ptr, offset_val);
                            } else {
                                // For scalar fields, load the value
                                return self.builder.ins().load(
                                    types::I64,
                                    cranelift::prelude::MemFlags::new(),
                                    ptr,
                                    field_info.offset as i32,
                                );
                            }
                        }
                    }
                }
                // Fallback: search all layouts (for cases where type inference fails)
                for layout in self.struct_layouts.values() {
                    if let Some(field_info) = layout.fields.get(field) {
                        let is_struct_field = self.struct_layouts.contains_key(&field_info.type_name);
                        if is_struct_field {
                            let offset_val = self.builder.ins().iconst(types::I64, field_info.offset as i64);
                            return self.builder.ins().iadd(ptr, offset_val);
                        } else {
                            return self.builder.ins().load(
                                types::I64,
                                cranelift::prelude::MemFlags::new(),
                                ptr,
                                field_info.offset as i32,
                            );
                        }
                    }
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Array(elems) => {
                if elems.is_empty() {
                    return self.builder.ins().iconst(types::I64, 0);
                }
                // Allocate stack slot for array (8 bytes per element)
                let size = elems.len() * 8;
                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                    size as u32,
                    0,
                ));
                let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                
                // Store each element
                for (i, elem) in elems.iter().enumerate() {
                    let val = self.translate_expr(elem);
                    self.builder.ins().store(
                        cranelift::prelude::MemFlags::new(),
                        val,
                        ptr,
                        (i * 8) as i32,
                    );
                }
                
                ptr
            }
            ExprKind::Index { expr: base, index } => {
                let base_type = self.infer_type_name(base);
                let idx = self.translate_expr(index);

                // Check if base is a Vec or String (BoltVec layout)
                let is_vec = base_type.as_ref()
                    .map(|t| t == "Vec" || t.starts_with("Vec<") || t == "String")
                    .unwrap_or(false);

                if is_vec {
                    // Vec/String indexing with bounds check
                    let vec_ptr = self.translate_expr(base);

                    // Load data pointer (offset 0) and length (offset 8) from Vec struct
                    let data_ptr = self.builder.ins().load(
                        types::I64,
                        cranelift::prelude::MemFlags::new(),
                        vec_ptr,
                        0,
                    );
                    let len = self.builder.ins().load(
                        types::I64,
                        cranelift::prelude::MemFlags::new(),
                        vec_ptr,
                        8,
                    );

                    // Bounds check: if idx >= len, panic
                    let in_bounds = self.builder.ins().icmp(IntCC::UnsignedLessThan, idx, len);

                    let ok_block = self.builder.create_block();
                    let panic_block = self.builder.create_block();

                    self.builder.ins().brif(in_bounds, ok_block, &[], panic_block, &[]);

                    // Panic block - call bolt_panic
                    self.builder.switch_to_block(panic_block);
                    self.builder.seal_block(panic_block);

                    // Create panic message on stack
                    let panic_msg = b"index out of bounds";
                    let msg_len = panic_msg.len();
                    let msg_slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                        msg_len as u32,
                        0,
                    ));
                    let msg_ptr = self.builder.ins().stack_addr(types::I64, msg_slot, 0);
                    for (i, &byte) in panic_msg.iter().enumerate() {
                        let byte_val = self.builder.ins().iconst(types::I8, byte as i64);
                        self.builder.ins().store(cranelift::prelude::MemFlags::new(), byte_val, msg_ptr, i as i32);
                    }
                    let msg_len_val = self.builder.ins().iconst(types::I64, msg_len as i64);

                    if let Some(&panic_fn) = self.func_names.get("bolt_panic") {
                        let panic_ref = self.module.declare_func_in_func(panic_fn, self.builder.func);
                        self.builder.ins().call(panic_ref, &[msg_ptr, msg_len_val]);
                    }
                    self.builder.ins().trap(cranelift::prelude::TrapCode::User(0));

                    // OK block - do the actual indexing
                    self.builder.switch_to_block(ok_block);
                    self.builder.seal_block(ok_block);

                    // Calculate address: data_ptr + idx * elem_size
                    let elem_size = self.builder.ins().iconst(types::I64, 8);
                    let offset = self.builder.ins().imul(idx, elem_size);
                    let addr = self.builder.ins().iadd(data_ptr, offset);

                    // Load and return the value
                    self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), addr, 0)
                } else {
                    // Check if base is a slice local (has companion _ptr and _len)
                    if let ExprKind::Path(path) = &base.kind {
                        if path.segments.len() == 1 {
                            let name = &path.segments[0].ident;
                            let ptr_name = format!("{}_ptr", name);
                            let len_name = format!("{}_len", name);

                            if let (Some(ptr_info), Some(len_info)) = (self.locals.get(&ptr_name).cloned(), self.locals.get(&len_name).cloned()) {
                                // Slice indexing with bounds check
                                let data_ptr = self.builder.use_var(ptr_info.var);
                                let len = self.builder.use_var(len_info.var);

                                // Bounds check
                                let in_bounds = self.builder.ins().icmp(IntCC::UnsignedLessThan, idx, len);
                                let ok_block = self.builder.create_block();
                                let panic_block = self.builder.create_block();

                                self.builder.ins().brif(in_bounds, ok_block, &[], panic_block, &[]);

                                // Panic block
                                self.builder.switch_to_block(panic_block);
                                self.builder.seal_block(panic_block);
                                if let Some(&abort_fn) = self.func_names.get("bolt_abort") {
                                    let abort_ref = self.module.declare_func_in_func(abort_fn, self.builder.func);
                                    self.builder.ins().call(abort_ref, &[]);
                                }
                                self.builder.ins().trap(cranelift::prelude::TrapCode::User(0));

                                // OK block - do the indexing
                                self.builder.switch_to_block(ok_block);
                                self.builder.seal_block(ok_block);

                                let offset = self.builder.ins().imul_imm(idx, 8);
                                let addr = self.builder.ins().iadd(data_ptr, offset);
                                return self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), addr, 0);
                            }
                        }
                    }

                    // Raw array indexing (no bounds check, direct pointer arithmetic)
                    let ptr = self.translate_expr(base);
                    let eight = self.builder.ins().iconst(types::I64, 8);
                    let offset = self.builder.ins().imul(idx, eight);
                    let addr = self.builder.ins().iadd(ptr, offset);
                    self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), addr, 0)
                }
            }
            ExprKind::Match { expr: scrutinee, arms } => {
                let scrutinee_val = self.translate_expr(scrutinee);
                
                let merge_block = self.builder.create_block();
                self.builder.append_block_param(merge_block, types::I64);
                
                // Create blocks for each arm
                let arm_blocks: Vec<Block> = arms.iter().map(|_| self.builder.create_block()).collect();
                let fail_block = self.builder.create_block();
                
                // Jump to first arm test
                if !arm_blocks.is_empty() {
                    self.builder.ins().jump(arm_blocks[0], &[]);
                } else {
                    self.builder.ins().jump(fail_block, &[]);
                }
                
                for (i, arm) in arms.iter().enumerate() {
                    self.builder.switch_to_block(arm_blocks[i]);
                    self.builder.seal_block(arm_blocks[i]);
                    
                    let next_block = if i + 1 < arm_blocks.len() {
                        arm_blocks[i + 1]
                    } else {
                        fail_block
                    };
                    
                    // Check pattern
                    match &arm.pattern.kind {
                        PatternKind::Wild => {
                            // Wildcard matches everything
                            let result = self.translate_expr(&arm.body);
                            self.builder.ins().jump(merge_block, &[result]);
                        }
                        PatternKind::Lit(lit) => {
                            let pat_val = self.translate_literal(lit);
                            let cmp = self.builder.ins().icmp(IntCC::Equal, scrutinee_val, pat_val);

                            let body_block = self.builder.create_block();
                            self.builder.ins().brif(cmp, body_block, &[], next_block, &[]);

                            self.builder.switch_to_block(body_block);
                            self.builder.seal_block(body_block);
                            let result = self.translate_expr(&arm.body);
                            self.builder.ins().jump(merge_block, &[result]);
                        }
                        PatternKind::Ident { name, .. } => {
                            // Binding pattern - bind value and execute body
                            let var = Variable::from_u32(self.next_var as u32);
                            self.next_var += 1;
                            self.builder.declare_var(var, types::I64);
                            self.builder.def_var(var, scrutinee_val);
                            self.locals.insert(name.clone(), LocalInfo { var, ty: types::I64 });

                            let result = self.translate_expr(&arm.body);
                            self.builder.ins().jump(merge_block, &[result]);
                        }
                        PatternKind::Path(path) => {
                            // Enum unit variant pattern: MyEnum::Variant
                            if path.segments.len() == 2 {
                                let enum_name = &path.segments[0].ident;
                                let variant_name = &path.segments[1].ident;
                                if let Some(enum_layout) = self.enum_layouts.get(enum_name) {
                                    if let Some(variant) = enum_layout.variants.get(variant_name) {
                                        // Load discriminant from enum pointer
                                        let disc = self.builder.ins().load(
                                            types::I64,
                                            cranelift::prelude::MemFlags::new(),
                                            scrutinee_val,
                                            0,
                                        );
                                        let expected = self.builder.ins().iconst(types::I64, variant.discriminant);
                                        let cmp = self.builder.ins().icmp(IntCC::Equal, disc, expected);

                                        let body_block = self.builder.create_block();
                                        self.builder.ins().brif(cmp, body_block, &[], next_block, &[]);

                                        self.builder.switch_to_block(body_block);
                                        self.builder.seal_block(body_block);
                                        let result = self.translate_expr(&arm.body);
                                        self.builder.ins().jump(merge_block, &[result]);
                                    } else {
                                        self.builder.ins().jump(next_block, &[]);
                                    }
                                } else {
                                    self.builder.ins().jump(next_block, &[]);
                                }
                            } else {
                                self.builder.ins().jump(next_block, &[]);
                            }
                        }
                        PatternKind::TupleStruct { path, elems } => {
                            // Enum tuple variant pattern: Some(x) or MyEnum::Variant(a, b)
                            let (enum_name, variant_name) = if path.segments.len() == 2 {
                                (&path.segments[0].ident, &path.segments[1].ident)
                            } else if path.segments.len() == 1 {
                                // Try to find variant in any enum (for Option::Some style)
                                let variant_name = &path.segments[0].ident;
                                let mut found = None;
                                for (ename, elayout) in self.enum_layouts.iter() {
                                    if elayout.variants.contains_key(variant_name) {
                                        found = Some((ename.clone(), variant_name.clone()));
                                        break;
                                    }
                                }
                                if let Some((en, vn)) = found {
                                    // We need owned strings, let's restructure
                                    if let Some(enum_layout) = self.enum_layouts.get(&en) {
                                        if let Some(variant) = enum_layout.variants.get(&vn) {
                                            // Load discriminant
                                            let disc = self.builder.ins().load(
                                                types::I64,
                                                cranelift::prelude::MemFlags::new(),
                                                scrutinee_val,
                                                0,
                                            );
                                            let expected = self.builder.ins().iconst(types::I64, variant.discriminant);
                                            let cmp = self.builder.ins().icmp(IntCC::Equal, disc, expected);

                                            let body_block = self.builder.create_block();
                                            self.builder.ins().brif(cmp, body_block, &[], next_block, &[]);

                                            self.builder.switch_to_block(body_block);
                                            self.builder.seal_block(body_block);

                                            // Bind tuple elements
                                            for (idx, elem_pat) in elems.iter().enumerate() {
                                                match &elem_pat.kind {
                                                    PatternKind::Ident { name, .. } => {
                                                        let val = self.builder.ins().load(
                                                            types::I64,
                                                            cranelift::prelude::MemFlags::new(),
                                                            scrutinee_val,
                                                            (8 + idx * 8) as i32,
                                                        );
                                                        let var = Variable::from_u32(self.next_var as u32);
                                                        self.next_var += 1;
                                                        self.builder.declare_var(var, types::I64);
                                                        self.builder.def_var(var, val);
                                                        self.locals.insert(name.clone(), LocalInfo { var, ty: types::I64 });
                                                    }
                                                    PatternKind::Struct { path, fields, .. } => {
                                                        // Level 2: nested struct pattern like Some(Point { x, y })
                                                        // Load the struct pointer from the enum payload
                                                        let struct_ptr = self.builder.ins().load(
                                                            types::I64,
                                                            cranelift::prelude::MemFlags::new(),
                                                            scrutinee_val,
                                                            (8 + idx * 8) as i32,
                                                        );
                                                        // Get struct name and layout
                                                        if let Some(seg) = path.segments.first() {
                                                            let struct_name = &seg.ident;
                                                            if let Some(layout) = self.struct_layouts.get(struct_name).cloned() {
                                                                // Bind each field
                                                                for field_pat in fields {
                                                                    if let Some(field_info) = layout.fields.get(&field_pat.name) {
                                                                        // Load field value
                                                                        let field_val = self.builder.ins().load(
                                                                            types::I64,
                                                                            cranelift::prelude::MemFlags::new(),
                                                                            struct_ptr,
                                                                            field_info.offset as i32,
                                                                        );
                                                                        // Bind to pattern - for now just support Ident
                                                                        if let PatternKind::Ident { name, .. } = &field_pat.pattern.kind {
                                                                            let var = Variable::from_u32(self.next_var as u32);
                                                                            self.next_var += 1;
                                                                            self.builder.declare_var(var, types::I64);
                                                                            self.builder.def_var(var, field_val);
                                                                            self.locals.insert(name.clone(), LocalInfo { var, ty: types::I64 });
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    _ => {}
                                                }
                                            }

                                            let result = self.translate_expr(&arm.body);
                                            self.builder.ins().jump(merge_block, &[result]);
                                        } else {
                                            self.builder.ins().jump(next_block, &[]);
                                        }
                                    } else {
                                        self.builder.ins().jump(next_block, &[]);
                                    }
                                    continue;
                                }
                                self.builder.ins().jump(next_block, &[]);
                                continue;
                            } else {
                                self.builder.ins().jump(next_block, &[]);
                                continue;
                            };

                            if let Some(enum_layout) = self.enum_layouts.get(enum_name) {
                                if let Some(variant) = enum_layout.variants.get(variant_name) {
                                    // Load discriminant
                                    let disc = self.builder.ins().load(
                                        types::I64,
                                        cranelift::prelude::MemFlags::new(),
                                        scrutinee_val,
                                        0,
                                    );
                                    let expected = self.builder.ins().iconst(types::I64, variant.discriminant);
                                    let cmp = self.builder.ins().icmp(IntCC::Equal, disc, expected);

                                    let body_block = self.builder.create_block();
                                    self.builder.ins().brif(cmp, body_block, &[], next_block, &[]);

                                    self.builder.switch_to_block(body_block);
                                    self.builder.seal_block(body_block);

                                    // Bind tuple elements
                                    for (idx, elem_pat) in elems.iter().enumerate() {
                                        match &elem_pat.kind {
                                            PatternKind::Ident { name, .. } => {
                                                let val = self.builder.ins().load(
                                                    types::I64,
                                                    cranelift::prelude::MemFlags::new(),
                                                    scrutinee_val,
                                                    (8 + idx * 8) as i32,
                                                );
                                                let var = Variable::from_u32(self.next_var as u32);
                                                self.next_var += 1;
                                                self.builder.declare_var(var, types::I64);
                                                self.builder.def_var(var, val);
                                                self.locals.insert(name.clone(), LocalInfo { var, ty: types::I64 });
                                            }
                                            PatternKind::Struct { path, fields, .. } => {
                                                // Level 2: nested struct pattern
                                                let struct_ptr = self.builder.ins().load(
                                                    types::I64,
                                                    cranelift::prelude::MemFlags::new(),
                                                    scrutinee_val,
                                                    (8 + idx * 8) as i32,
                                                );
                                                if let Some(seg) = path.segments.first() {
                                                    let struct_name = &seg.ident;
                                                    if let Some(layout) = self.struct_layouts.get(struct_name).cloned() {
                                                        for field_pat in fields {
                                                            if let Some(field_info) = layout.fields.get(&field_pat.name) {
                                                                let field_val = self.builder.ins().load(
                                                                    types::I64,
                                                                    cranelift::prelude::MemFlags::new(),
                                                                    struct_ptr,
                                                                    field_info.offset as i32,
                                                                );
                                                                if let PatternKind::Ident { name, .. } = &field_pat.pattern.kind {
                                                                    let var = Variable::from_u32(self.next_var as u32);
                                                                    self.next_var += 1;
                                                                    self.builder.declare_var(var, types::I64);
                                                                    self.builder.def_var(var, field_val);
                                                                    self.locals.insert(name.clone(), LocalInfo { var, ty: types::I64 });
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }
                                    }

                                    let result = self.translate_expr(&arm.body);
                                    self.builder.ins().jump(merge_block, &[result]);
                                } else {
                                    self.builder.ins().jump(next_block, &[]);
                                }
                            } else {
                                self.builder.ins().jump(next_block, &[]);
                            }
                        }
                        _ => {
                            // Unsupported pattern, try next arm
                            self.builder.ins().jump(next_block, &[]);
                        }
                    }
                }
                
                // Fail block returns 0
                self.builder.switch_to_block(fail_block);
                self.builder.seal_block(fail_block);
                let zero = self.builder.ins().iconst(types::I64, 0);
                self.builder.ins().jump(merge_block, &[zero]);
                
                self.builder.switch_to_block(merge_block);
                self.builder.seal_block(merge_block);
                self.builder.block_params(merge_block)[0]
            }
            ExprKind::Range { lo, hi, inclusive } => {
                // For simple integer ranges, store (current, end) on stack
                // This is used by for loops
                let start = lo.as_ref()
                    .map(|e| self.translate_expr(e))
                    .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
                let end = hi.as_ref()
                    .map(|e| self.translate_expr(e))
                    .unwrap_or_else(|| self.builder.ins().iconst(types::I64, i64::MAX));
                
                // Allocate stack slot for range (16 bytes: start + end)
                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                    16,
                    0,
                ));
                let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                
                // Store start and end
                self.builder.ins().store(cranelift::prelude::MemFlags::new(), start, ptr, 0);
                let end_val = if *inclusive {
                    self.builder.ins().iadd_imm(end, 1)
                } else {
                    end
                };
                self.builder.ins().store(cranelift::prelude::MemFlags::new(), end_val, ptr, 8);
                
                ptr
            }
            ExprKind::Ref { mutable: _, expr: inner } => {
                // Take address of expression
                // For variables, we need to handle differently based on type
                if let ExprKind::Path(path) = &inner.kind {
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident;
                        if let Some(info) = self.locals.get(name) {
                            let val = self.builder.use_var(info.var);

                            // Check if this is an array/struct that already holds a pointer
                            // Arrays are stored as pointers to stack-allocated data
                            if self.local_array_lens.contains_key(name) {
                                // Array: the value IS already a pointer to the array data
                                return val;
                            }

                            // Check if it's a struct type (stored as pointer)
                            if let Some(type_name) = self.local_types.get(name) {
                                if self.struct_layouts.contains_key(type_name) {
                                    // Struct: the value IS already a pointer
                                    return val;
                                }
                            }

                            // For scalar values, allocate stack slot and store value
                            let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                                cranelift::prelude::StackSlotKind::ExplicitSlot,
                                8,
                                0,
                            ));
                            let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                            self.builder.ins().store(cranelift::prelude::MemFlags::new(), val, ptr, 0);
                            return ptr;
                        }
                    }
                }
                // For array literals, just return the pointer directly
                if let ExprKind::Array(_) = &inner.kind {
                    return self.translate_expr(inner);
                }
                // For other expressions, translate and store on stack
                let val = self.translate_expr(inner);
                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                    8,
                    0,
                ));
                let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                self.builder.ins().store(cranelift::prelude::MemFlags::new(), val, ptr, 0);
                ptr
            }
            ExprKind::Deref(inner) => {
                // Dereference pointer - load from address
                let ptr = self.translate_expr(inner);
                self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), ptr, 0)
            }
            ExprKind::For { pattern, iter, body, .. } => {
                // Desugar: for i in start..end { body }
                // becomes: let mut _i = start; while _i < end { let i = _i; body; _i += 1; }

                // Check if this is a Vec/array iterator (.iter() call)
                let (is_vec_iter, is_array_iter, array_len) = if let ExprKind::MethodCall { receiver, method, .. } = &iter.kind {
                    if method == "iter" {
                        let recv_type = self.infer_type_name(receiver);
                        let is_vec = recv_type.as_ref().map(|t| t == "Vec" || t.starts_with("Vec<")).unwrap_or(false);

                        // Check if receiver is an array expression or array variable
                        let (is_arr, arr_len) = if let ExprKind::Array(elems) = &receiver.kind {
                            (true, elems.len() as i64)
                        } else {
                            // Could be an array variable - check local_types
                            // For now, just return false
                            (false, 0)
                        };

                        (is_vec, is_arr, arr_len)
                    } else {
                        (false, false, 0)
                    }
                } else {
                    (false, false, 0)
                };

                if is_vec_iter {
                    // Handle Vec iterator: for x in vec.iter() { ... }
                    // Get the receiver (the Vec)
                    if let ExprKind::MethodCall { receiver, .. } = &iter.kind {
                        let vec_ptr = self.translate_expr(receiver);

                        // Vec layout: data_ptr(0), len(8), cap(16)
                        let data_ptr = self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), vec_ptr, 0);
                        let len = self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), vec_ptr, 8);

                        // Create index counter (0 to len)
                        let counter_var = Variable::from_u32(self.next_var as u32);
                        self.next_var += 1;
                        self.builder.declare_var(counter_var, types::I64);
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        self.builder.def_var(counter_var, zero);

                        // Store len and data_ptr in variables
                        let len_var = Variable::from_u32(self.next_var as u32);
                        self.next_var += 1;
                        self.builder.declare_var(len_var, types::I64);
                        self.builder.def_var(len_var, len);

                        let data_var = Variable::from_u32(self.next_var as u32);
                        self.next_var += 1;
                        self.builder.declare_var(data_var, types::I64);
                        self.builder.def_var(data_var, data_ptr);

                        // Create element variable for the loop body
                        let elem_var = Variable::from_u32(self.next_var as u32);
                        self.next_var += 1;
                        self.builder.declare_var(elem_var, types::I64);
                        self.builder.def_var(elem_var, zero);

                        // Bind pattern name to element variable
                        if let PatternKind::Ident { name, .. } = &pattern.kind {
                            self.locals.insert(name.clone(), LocalInfo { var: elem_var, ty: types::I64 });
                        }

                        let header_block = self.builder.create_block();
                        let body_block = self.builder.create_block();
                        let exit_block = self.builder.create_block();

                        self.builder.ins().jump(header_block, &[]);

                        self.builder.switch_to_block(header_block);

                        let idx = self.builder.use_var(counter_var);
                        let len_val = self.builder.use_var(len_var);
                        let cond = self.builder.ins().icmp(IntCC::SignedLessThan, idx, len_val);
                        self.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

                        self.builder.switch_to_block(body_block);
                        self.builder.seal_block(body_block);

                        // Load element: data_ptr[idx * 8]
                        let data = self.builder.use_var(data_var);
                        let idx = self.builder.use_var(counter_var);
                        let offset = self.builder.ins().imul_imm(idx, 8);
                        let elem_addr = self.builder.ins().iadd(data, offset);
                        let elem_val = self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), elem_addr, 0);
                        self.builder.def_var(elem_var, elem_val);

                        self.loop_stack.push((header_block, exit_block));
                        self.translate_block(body);
                        self.loop_stack.pop();

                        // Increment counter
                        let idx = self.builder.use_var(counter_var);
                        let next_idx = self.builder.ins().iadd_imm(idx, 1);
                        self.builder.def_var(counter_var, next_idx);

                        self.builder.ins().jump(header_block, &[]);

                        self.builder.seal_block(header_block);

                        self.builder.switch_to_block(exit_block);
                        self.builder.seal_block(exit_block);

                        return self.builder.ins().iconst(types::I64, 0);
                    }
                }

                if is_array_iter {
                    // Handle array iterator: for x in [1,2,3].iter() { ... }
                    if let ExprKind::MethodCall { receiver, .. } = &iter.kind {
                        let array_ptr = self.translate_expr(receiver);

                        // Array is a contiguous block of memory, pointer is the start
                        let len = self.builder.ins().iconst(types::I64, array_len);

                        // Create index counter (0 to len)
                        let counter_var = Variable::from_u32(self.next_var as u32);
                        self.next_var += 1;
                        self.builder.declare_var(counter_var, types::I64);
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        self.builder.def_var(counter_var, zero);

                        // Store len and array_ptr in variables
                        let len_var = Variable::from_u32(self.next_var as u32);
                        self.next_var += 1;
                        self.builder.declare_var(len_var, types::I64);
                        self.builder.def_var(len_var, len);

                        let data_var = Variable::from_u32(self.next_var as u32);
                        self.next_var += 1;
                        self.builder.declare_var(data_var, types::I64);
                        self.builder.def_var(data_var, array_ptr);

                        // Create element variable for the loop body
                        let elem_var = Variable::from_u32(self.next_var as u32);
                        self.next_var += 1;
                        self.builder.declare_var(elem_var, types::I64);
                        self.builder.def_var(elem_var, zero);

                        // Bind pattern name to element variable
                        if let PatternKind::Ident { name, .. } = &pattern.kind {
                            self.locals.insert(name.clone(), LocalInfo { var: elem_var, ty: types::I64 });
                        }

                        let header_block = self.builder.create_block();
                        let body_block = self.builder.create_block();
                        let exit_block = self.builder.create_block();

                        self.builder.ins().jump(header_block, &[]);

                        self.builder.switch_to_block(header_block);

                        let idx = self.builder.use_var(counter_var);
                        let len_val = self.builder.use_var(len_var);
                        let cond = self.builder.ins().icmp(IntCC::SignedLessThan, idx, len_val);
                        self.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

                        self.builder.switch_to_block(body_block);
                        self.builder.seal_block(body_block);

                        // Load element: array_ptr[idx * 8]
                        let data = self.builder.use_var(data_var);
                        let idx = self.builder.use_var(counter_var);
                        let offset = self.builder.ins().imul_imm(idx, 8);
                        let elem_addr = self.builder.ins().iadd(data, offset);
                        let elem_val = self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), elem_addr, 0);
                        self.builder.def_var(elem_var, elem_val);

                        self.loop_stack.push((header_block, exit_block));
                        self.translate_block(body);
                        self.loop_stack.pop();

                        // Increment counter
                        let idx = self.builder.use_var(counter_var);
                        let next_idx = self.builder.ins().iadd_imm(idx, 1);
                        self.builder.def_var(counter_var, next_idx);

                        self.builder.ins().jump(header_block, &[]);

                        self.builder.seal_block(header_block);

                        self.builder.switch_to_block(exit_block);
                        self.builder.seal_block(exit_block);

                        return self.builder.ins().iconst(types::I64, 0);
                    }
                }

                // Extract start and end from range expression directly
                let (start, end) = if let ExprKind::Range { lo, hi, inclusive } = &iter.kind {
                    let s = lo.as_ref()
                        .map(|e| self.translate_expr(e))
                        .unwrap_or_else(|| self.builder.ins().iconst(types::I64, 0));
                    let e = hi.as_ref()
                        .map(|e| self.translate_expr(e))
                        .unwrap_or_else(|| self.builder.ins().iconst(types::I64, i64::MAX));
                    let e = if *inclusive {
                        self.builder.ins().iadd_imm(e, 1)
                    } else {
                        e
                    };
                    (s, e)
                } else {
                    // Fallback: treat iter as a range pointer
                    let range_ptr = self.translate_expr(iter);
                    let s = self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), range_ptr, 0);
                    let e = self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), range_ptr, 8);
                    (s, e)
                };

                // Create loop counter variable
                let counter_var = Variable::from_u32(self.next_var as u32);
                self.next_var += 1;
                self.builder.declare_var(counter_var, types::I64);
                self.builder.def_var(counter_var, start);

                // Store end in a variable so it's available across blocks
                let end_var = Variable::from_u32(self.next_var as u32);
                self.next_var += 1;
                self.builder.declare_var(end_var, types::I64);
                self.builder.def_var(end_var, end);

                // Bind pattern name to counter
                if let PatternKind::Ident { name, .. } = &pattern.kind {
                    self.locals.insert(name.clone(), LocalInfo { var: counter_var, ty: types::I64 });
                }

                let header_block = self.builder.create_block();
                let body_block = self.builder.create_block();
                let exit_block = self.builder.create_block();

                self.builder.ins().jump(header_block, &[]);

                self.builder.switch_to_block(header_block);
                // Don't seal yet

                let current = self.builder.use_var(counter_var);
                let end_val = self.builder.use_var(end_var);
                let cond = self.builder.ins().icmp(IntCC::SignedLessThan, current, end_val);
                self.builder.ins().brif(cond, body_block, &[], exit_block, &[]);
                
                self.builder.switch_to_block(body_block);
                self.builder.seal_block(body_block);
                
                self.loop_stack.push((header_block, exit_block));
                self.translate_block(body);
                self.loop_stack.pop();
                
                // Increment counter
                let current = self.builder.use_var(counter_var);
                let next = self.builder.ins().iadd_imm(current, 1);
                self.builder.def_var(counter_var, next);
                
                self.builder.ins().jump(header_block, &[]);
                
                self.builder.seal_block(header_block);
                
                self.builder.switch_to_block(exit_block);
                self.builder.seal_block(exit_block);
                
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Try(inner) => {
                // The ? operator: extract Ok value or return Err early
                // Result/Option are laid out as: [discriminant: i64, payload: i64]
                // Ok/Some = discriminant 0, Err/None = discriminant 1
                let result_ptr = self.translate_expr(inner);

                // Load discriminant
                let disc = self.builder.ins().load(
                    types::I64,
                    cranelift::prelude::MemFlags::new(),
                    result_ptr,
                    0,
                );

                // Check if Ok (discriminant == 0)
                let zero = self.builder.ins().iconst(types::I64, 0);
                let is_ok = self.builder.ins().icmp(IntCC::Equal, disc, zero);

                let ok_block = self.builder.create_block();
                let err_block = self.builder.create_block();
                let merge_block = self.builder.create_block();
                self.builder.append_block_param(merge_block, types::I64);

                self.builder.ins().brif(is_ok, ok_block, &[], err_block, &[]);

                // Ok path - extract the value and continue
                self.builder.switch_to_block(ok_block);
                self.builder.seal_block(ok_block);
                let ok_value = self.builder.ins().load(
                    types::I64,
                    cranelift::prelude::MemFlags::new(),
                    result_ptr,
                    8,  // payload at offset 8
                );
                self.builder.ins().jump(merge_block, &[ok_value]);

                // Err path - return early with the error wrapped in Err
                self.builder.switch_to_block(err_block);
                self.builder.seal_block(err_block);

                // Allocate a new Result for the error return
                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                    16,  // discriminant + payload
                    0,
                ));
                let err_result = self.builder.ins().stack_addr(types::I64, slot, 0);

                // Copy the original error to the new Result
                let err_value = self.builder.ins().load(
                    types::I64,
                    cranelift::prelude::MemFlags::new(),
                    result_ptr,
                    8,
                );
                let one = self.builder.ins().iconst(types::I64, 1);  // Err discriminant
                self.builder.ins().store(cranelift::prelude::MemFlags::new(), one, err_result, 0);
                self.builder.ins().store(cranelift::prelude::MemFlags::new(), err_value, err_result, 8);

                // Return early with the error
                self.builder.ins().return_(&[err_result]);

                self.builder.switch_to_block(merge_block);
                self.builder.seal_block(merge_block);

                self.builder.block_params(merge_block)[0]
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
            Literal::Str(s) => {
                // String literal: allocate on stack, store bytes, return (ptr, len) packed
                // For simplicity, just return ptr. Caller should know the length.
                let bytes = s.as_bytes();
                let len = bytes.len();
                if len == 0 {
                    return self.builder.ins().iconst(types::I64, 0);
                }
                // Allocate stack slot for string data
                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                    len as u32,
                    0,
                ));
                let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                // Store each byte
                for (i, &byte) in bytes.iter().enumerate() {
                    let val = self.builder.ins().iconst(types::I8, byte as i64);
                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), val, ptr, i as i32);
                }
                ptr
            }
            Literal::ByteStr(bytes) => {
                let len = bytes.len();
                if len == 0 {
                    return self.builder.ins().iconst(types::I64, 0);
                }
                let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                    cranelift::prelude::StackSlotKind::ExplicitSlot,
                    len as u32,
                    0,
                ));
                let ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                for (i, &byte) in bytes.iter().enumerate() {
                    let val = self.builder.ins().iconst(types::I8, byte as i64);
                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), val, ptr, i as i32);
                }
                ptr
            }
        }
    }

    /// Coerce a value to a target type (handles integer width conversions)
    fn coerce_to_type(&mut self, val: Value, target: types::Type) -> Value {
        let val_ty = self.builder.func.dfg.value_type(val);
        if val_ty == target {
            return val;
        }

        // Handle integer to integer conversions
        if val_ty.is_int() && target.is_int() {
            let val_bits = val_ty.bits();
            let target_bits = target.bits();

            if val_bits > target_bits {
                // Reduce: larger to smaller (e.g., i64 -> i32)
                return self.builder.ins().ireduce(target, val);
            } else {
                // Extend: smaller to larger (e.g., i32 -> i64)
                return self.builder.ins().sextend(target, val);
            }
        }

        // For other types, just return as-is
        val
    }

    fn translate_binop(&mut self, op: BinaryOp, lhs: Value, rhs: Value) -> Value {
        // Check if operands are floats by inspecting the value type
        let lhs_ty = self.builder.func.dfg.value_type(lhs);
        let rhs_ty = self.builder.func.dfg.value_type(rhs);
        let is_float = lhs_ty == types::F32 || lhs_ty == types::F64;

        // Unify integer types: coerce both to the larger type
        let (lhs, rhs) = if !is_float && lhs_ty != rhs_ty && lhs_ty.is_int() && rhs_ty.is_int() {
            let target = if lhs_ty.bits() > rhs_ty.bits() { lhs_ty } else { rhs_ty };
            let lhs_coerced = self.coerce_to_type(lhs, target);
            let rhs_coerced = self.coerce_to_type(rhs, target);
            (lhs_coerced, rhs_coerced)
        } else {
            (lhs, rhs)
        };

        match op {
            BinaryOp::Add => {
                if is_float {
                    self.builder.ins().fadd(lhs, rhs)
                } else {
                    self.builder.ins().iadd(lhs, rhs)
                }
            }
            BinaryOp::Sub => {
                if is_float {
                    self.builder.ins().fsub(lhs, rhs)
                } else {
                    self.builder.ins().isub(lhs, rhs)
                }
            }
            BinaryOp::Mul => {
                if is_float {
                    self.builder.ins().fmul(lhs, rhs)
                } else {
                    self.builder.ins().imul(lhs, rhs)
                }
            }
            BinaryOp::Div => {
                if is_float {
                    self.builder.ins().fdiv(lhs, rhs)
                } else {
                    self.builder.ins().sdiv(lhs, rhs)
                }
            }
            BinaryOp::Rem => {
                // Float remainder not directly supported, use integer
                self.builder.ins().srem(lhs, rhs)
            }
            BinaryOp::BitAnd => self.builder.ins().band(lhs, rhs),
            BinaryOp::BitOr => self.builder.ins().bor(lhs, rhs),
            BinaryOp::BitXor => self.builder.ins().bxor(lhs, rhs),
            BinaryOp::Shl => self.builder.ins().ishl(lhs, rhs),
            BinaryOp::Shr => self.builder.ins().sshr(lhs, rhs),
            BinaryOp::Eq => {
                if is_float {
                    self.builder.ins().fcmp(cranelift::prelude::FloatCC::Equal, lhs, rhs)
                } else {
                    self.builder.ins().icmp(IntCC::Equal, lhs, rhs)
                }
            }
            BinaryOp::Ne => {
                if is_float {
                    self.builder.ins().fcmp(cranelift::prelude::FloatCC::NotEqual, lhs, rhs)
                } else {
                    self.builder.ins().icmp(IntCC::NotEqual, lhs, rhs)
                }
            }
            BinaryOp::Lt => {
                if is_float {
                    self.builder.ins().fcmp(cranelift::prelude::FloatCC::LessThan, lhs, rhs)
                } else {
                    self.builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs)
                }
            }
            BinaryOp::Le => {
                if is_float {
                    self.builder.ins().fcmp(cranelift::prelude::FloatCC::LessThanOrEqual, lhs, rhs)
                } else {
                    self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs)
                }
            }
            BinaryOp::Gt => {
                if is_float {
                    self.builder.ins().fcmp(cranelift::prelude::FloatCC::GreaterThan, lhs, rhs)
                } else {
                    self.builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs)
                }
            }
            BinaryOp::Ge => {
                if is_float {
                    self.builder.ins().fcmp(cranelift::prelude::FloatCC::GreaterThanOrEqual, lhs, rhs)
                } else {
                    self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs)
                }
            }
            BinaryOp::And | BinaryOp::Or => self.builder.ins().band(lhs, rhs),
        }
    }

    /// Translate Vec method calls to runtime function calls
    fn translate_vec_method(&mut self, recv_ptr: Value, method: &str, args: &[Expr]) -> Value {
        // recv_ptr is a pointer to a BoltVec struct (24 bytes: ptr, len, cap)
        // Element size is 8 bytes (i64) for now - we'd need type info for proper sizing
        let elem_size = self.builder.ins().iconst(types::I64, 8);

        match method {
            "len" => {
                // Call bolt_vec_len(vec_ptr) -> usize
                let func_id = self.func_names.get("bolt_vec_len")
                    .expect("bolt_vec_len not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[recv_ptr]);
                self.builder.inst_results(call)[0]
            }
            "capacity" => {
                // Call bolt_vec_capacity(vec_ptr) -> usize
                let func_id = self.func_names.get("bolt_vec_capacity")
                    .expect("bolt_vec_capacity not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[recv_ptr]);
                self.builder.inst_results(call)[0]
            }
            "is_empty" => {
                // Call bolt_vec_is_empty(vec_ptr) -> bool
                let func_id = self.func_names.get("bolt_vec_is_empty")
                    .expect("bolt_vec_is_empty not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[recv_ptr]);
                self.builder.inst_results(call)[0]
            }
            "push" => {
                // Call bolt_vec_push(vec_ptr, elem_ptr, elem_size)
                // First, evaluate the argument and store it on stack to get a pointer
                if let Some(arg) = args.first() {
                    let val = self.translate_expr(arg);
                    // Allocate stack slot for element
                    let slot = self.builder.create_sized_stack_slot(cranelift::prelude::StackSlotData::new(
                        cranelift::prelude::StackSlotKind::ExplicitSlot,
                        8,
                        0,
                    ));
                    let elem_ptr = self.builder.ins().stack_addr(types::I64, slot, 0);
                    self.builder.ins().store(cranelift::prelude::MemFlags::new(), val, elem_ptr, 0);

                    let func_id = self.func_names.get("bolt_vec_push")
                        .expect("bolt_vec_push not registered");
                    let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                    self.builder.ins().call(func_ref, &[recv_ptr, elem_ptr, elem_size]);
                }
                // push returns ()
                self.builder.ins().iconst(types::I64, 0)
            }
            "pop" => {
                // Call bolt_vec_pop(vec_ptr, elem_size) -> *mut u8
                let func_id = self.func_names.get("bolt_vec_pop")
                    .expect("bolt_vec_pop not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[recv_ptr, elem_size]);
                let result_ptr = self.builder.inst_results(call)[0];
                // Load the value from the returned pointer (if not null)
                // For simplicity, just load - caller should handle null check
                self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), result_ptr, 0)
            }
            "clear" => {
                // Call bolt_vec_clear(vec_ptr)
                let func_id = self.func_names.get("bolt_vec_clear")
                    .expect("bolt_vec_clear not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                self.builder.ins().call(func_ref, &[recv_ptr]);
                self.builder.ins().iconst(types::I64, 0)
            }
            "get" => {
                // Call bolt_vec_get(vec_ptr, index, elem_size) -> *const u8
                if let Some(idx_expr) = args.first() {
                    let idx = self.translate_expr(idx_expr);
                    let func_id = self.func_names.get("bolt_vec_get")
                        .expect("bolt_vec_get not registered");
                    let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                    let call = self.builder.ins().call(func_ref, &[recv_ptr, idx, elem_size]);
                    let elem_ptr = self.builder.inst_results(call)[0];
                    // Load value from pointer
                    self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), elem_ptr, 0)
                } else {
                    self.builder.ins().iconst(types::I64, 0)
                }
            }
            "iter" => {
                // For now, just return the vec pointer - iterator will be handled by for loop
                recv_ptr
            }
            "sum" => {
                // Call bolt_vec_sum_i64(vec_ptr) -> i64
                let func_id = self.func_names.get("bolt_vec_sum_i64")
                    .expect("bolt_vec_sum_i64 not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[recv_ptr]);
                self.builder.inst_results(call)[0]
            }
            _ => {
                // Unknown method - return 0
                self.builder.ins().iconst(types::I64, 0)
            }
        }
    }

    /// Translate String method calls - String has same layout as Vec (ptr, len, cap)
    fn translate_string_method(&mut self, recv_ptr: Value, method: &str, args: &[Expr]) -> Value {
        match method {
            "len" => {
                // String layout: ptr(0), len(8), cap(16) - load len directly
                self.builder.ins().load(
                    types::I64,
                    cranelift::prelude::MemFlags::new(),
                    recv_ptr,
                    8,
                )
            }
            "is_empty" => {
                let len = self.builder.ins().load(
                    types::I64,
                    cranelift::prelude::MemFlags::new(),
                    recv_ptr,
                    8,
                );
                let zero = self.builder.ins().iconst(types::I64, 0);
                self.builder.ins().icmp(IntCC::Equal, len, zero)
            }
            "as_ptr" => {
                let func_id = self.func_names.get("bolt_string_as_ptr")
                    .expect("bolt_string_as_ptr not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[recv_ptr]);
                self.builder.inst_results(call)[0]
            }
            "push" => {
                // push a char (byte for now)
                if let Some(arg) = args.first() {
                    let byte_val = self.translate_expr(arg);
                    let func_id = self.func_names.get("bolt_string_push_byte")
                        .expect("bolt_string_push_byte not registered");
                    let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                    self.builder.ins().call(func_ref, &[recv_ptr, byte_val]);
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            "push_str" => {
                // push_str takes a &str (ptr, len)
                if let Some(arg) = args.first() {
                    let str_ptr = self.translate_expr(arg);
                    // For string literals, we need to extract ptr and len
                    // For now, assume it's a stack-allocated string
                    if let ExprKind::Lit(Literal::Str(s)) = &arg.kind {
                        let len = self.builder.ins().iconst(types::I64, s.len() as i64);
                        let func_id = self.func_names.get("bolt_string_push_str")
                            .expect("bolt_string_push_str not registered");
                        let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                        self.builder.ins().call(func_ref, &[recv_ptr, str_ptr, len]);
                    }
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            _ => self.builder.ins().iconst(types::I64, 0)
        }
    }

    /// Translate slice method calls
    /// Slices are stored as (ptr, len) with companion locals {name}_ptr and {name}_len
    fn translate_slice_method(&mut self, receiver: &Expr, method: &str, args: &[Expr]) -> Value {
        // Get the slice variable name from the receiver
        let slice_name = if let ExprKind::Path(path) = &receiver.kind {
            if path.segments.len() == 1 {
                path.segments[0].ident.clone()
            } else {
                return self.builder.ins().iconst(types::I64, 0);
            }
        } else {
            return self.builder.ins().iconst(types::I64, 0);
        };

        let ptr_name = format!("{}_ptr", slice_name);
        let len_name = format!("{}_len", slice_name);

        match method {
            "len" => {
                // Return the length directly from the _len local
                if let Some(len_info) = self.locals.get(&len_name) {
                    return self.builder.use_var(len_info.var);
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            "is_empty" => {
                if let Some(len_info) = self.locals.get(&len_name) {
                    let len = self.builder.use_var(len_info.var);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    return self.builder.ins().icmp(IntCC::Equal, len, zero);
                }
                self.builder.ins().iconst(types::I8, 1) // Default to true (empty)
            }
            "first" => {
                // Return pointer to first element (slice_ptr + 0), or panic if empty
                if let Some(ptr_info) = self.locals.get(&ptr_name) {
                    let ptr = self.builder.use_var(ptr_info.var);
                    // Load first element
                    return self.builder.ins().load(
                        types::I64,
                        cranelift::prelude::MemFlags::new(),
                        ptr,
                        0,
                    );
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            "get" => {
                // get(index) - returns Option<&T> but we'll return the element or 0
                if let Some(idx_expr) = args.first() {
                    let idx = self.translate_expr(idx_expr);
                    if let (Some(ptr_info), Some(len_info)) = (self.locals.get(&ptr_name), self.locals.get(&len_name)) {
                        let ptr = self.builder.use_var(ptr_info.var);
                        let len = self.builder.use_var(len_info.var);

                        // Bounds check
                        let in_bounds = self.builder.ins().icmp(IntCC::UnsignedLessThan, idx, len);
                        let ok_block = self.builder.create_block();
                        let fail_block = self.builder.create_block();
                        let merge_block = self.builder.create_block();

                        self.builder.append_block_param(merge_block, types::I64);
                        self.builder.ins().brif(in_bounds, ok_block, &[], fail_block, &[]);

                        // OK: load element
                        self.builder.switch_to_block(ok_block);
                        self.builder.seal_block(ok_block);
                        let offset = self.builder.ins().imul_imm(idx, 8);
                        let elem_ptr = self.builder.ins().iadd(ptr, offset);
                        let elem = self.builder.ins().load(
                            types::I64,
                            cranelift::prelude::MemFlags::new(),
                            elem_ptr,
                            0,
                        );
                        self.builder.ins().jump(merge_block, &[elem]);

                        // Fail: return 0
                        self.builder.switch_to_block(fail_block);
                        self.builder.seal_block(fail_block);
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        self.builder.ins().jump(merge_block, &[zero]);

                        self.builder.switch_to_block(merge_block);
                        self.builder.seal_block(merge_block);
                        return self.builder.block_params(merge_block)[0];
                    }
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            _ => self.builder.ins().iconst(types::I64, 0)
        }
    }

    /// Translate HashMap method calls
    /// HashMap layout: entries_ptr(8) + len(8) + cap(8) = 24 bytes
    fn translate_hashmap_method(&mut self, recv_ptr: Value, method: &str, args: &[Expr]) -> Value {
        match method {
            "insert" => {
                // insert(key, value) - both i64 for now
                if args.len() >= 2 {
                    let key = self.translate_expr(&args[0]);
                    let value = self.translate_expr(&args[1]);
                    let func_id = self.func_names.get("bolt_hashmap_insert")
                        .expect("bolt_hashmap_insert not registered");
                    let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                    self.builder.ins().call(func_ref, &[recv_ptr, key, value]);
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            "get" => {
                // get(key) -> Option<&V> (returns pointer to value, or null)
                if let Some(key_expr) = args.first() {
                    let key = self.translate_expr(key_expr);
                    let func_id = self.func_names.get("bolt_hashmap_get")
                        .expect("bolt_hashmap_get not registered");
                    let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                    let call = self.builder.ins().call(func_ref, &[recv_ptr, key]);
                    let ptr = self.builder.inst_results(call)[0];

                    // If pointer is not null, load the value
                    // For simplicity, just load - caller should check for null
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let is_null = self.builder.ins().icmp(IntCC::Equal, ptr, zero);

                    let some_block = self.builder.create_block();
                    let none_block = self.builder.create_block();
                    let merge_block = self.builder.create_block();
                    self.builder.append_block_param(merge_block, types::I64);

                    self.builder.ins().brif(is_null, none_block, &[], some_block, &[]);

                    // Some block - load value
                    self.builder.switch_to_block(some_block);
                    self.builder.seal_block(some_block);
                    let val = self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), ptr, 0);
                    self.builder.ins().jump(merge_block, &[val]);

                    // None block - return 0
                    self.builder.switch_to_block(none_block);
                    self.builder.seal_block(none_block);
                    self.builder.ins().jump(merge_block, &[zero]);

                    self.builder.switch_to_block(merge_block);
                    self.builder.seal_block(merge_block);
                    return self.builder.block_params(merge_block)[0];
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            "contains_key" => {
                if let Some(key_expr) = args.first() {
                    let key = self.translate_expr(key_expr);
                    let func_id = self.func_names.get("bolt_hashmap_contains_key")
                        .expect("bolt_hashmap_contains_key not registered");
                    let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                    let call = self.builder.ins().call(func_ref, &[recv_ptr, key]);
                    return self.builder.inst_results(call)[0];
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            "remove" => {
                if let Some(key_expr) = args.first() {
                    let key = self.translate_expr(key_expr);
                    let func_id = self.func_names.get("bolt_hashmap_remove")
                        .expect("bolt_hashmap_remove not registered");
                    let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                    let call = self.builder.ins().call(func_ref, &[recv_ptr, key]);
                    return self.builder.inst_results(call)[0];
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            "len" => {
                let func_id = self.func_names.get("bolt_hashmap_len")
                    .expect("bolt_hashmap_len not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[recv_ptr]);
                self.builder.inst_results(call)[0]
            }
            "is_empty" => {
                let func_id = self.func_names.get("bolt_hashmap_is_empty")
                    .expect("bolt_hashmap_is_empty not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[recv_ptr]);
                self.builder.inst_results(call)[0]
            }
            "clear" => {
                let func_id = self.func_names.get("bolt_hashmap_clear")
                    .expect("bolt_hashmap_clear not registered");
                let func_ref = self.module.declare_func_in_func(*func_id, self.builder.func);
                self.builder.ins().call(func_ref, &[recv_ptr]);
                self.builder.ins().iconst(types::I64, 0)
            }
            _ => self.builder.ins().iconst(types::I64, 0)
        }
    }
}

// Runtime functions for heap allocation
extern "C" fn bolt_malloc(size: i64) -> i64 {
    let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
    unsafe { std::alloc::alloc(layout) as i64 }
}

extern "C" fn bolt_free(ptr: i64, size: i64) {
    if ptr != 0 {
        let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) }
    }
}

extern "C" fn bolt_realloc(ptr: i64, old_size: i64, new_size: i64) -> i64 {
    if ptr == 0 {
        return bolt_malloc(new_size);
    }
    let old_layout = std::alloc::Layout::from_size_align(old_size as usize, 8).unwrap();
    unsafe { std::alloc::realloc(ptr as *mut u8, old_layout, new_size as usize) as i64 }
}

extern "C" fn bolt_print_int(value: i64) {
    println!("{}", value);
}

extern "C" fn bolt_print_str(ptr: i64, len: i64) {
    if ptr != 0 && len > 0 {
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
        if let Ok(s) = std::str::from_utf8(slice) {
            print!("{}", s);
        }
    }
}
