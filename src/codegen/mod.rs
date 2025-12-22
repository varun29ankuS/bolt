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
use cranelift_module::{FuncId, Linkage, Module};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Field offset information for a struct
#[derive(Clone)]
struct StructLayout {
    fields: IndexMap<String, usize>,  // field name -> offset
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
        // bolt_malloc(size: i64) -> i64 (pointer)
        let mut malloc_sig = self.module.make_signature();
        malloc_sig.params.push(AbiParam::new(types::I64));
        malloc_sig.returns.push(AbiParam::new(types::I64));
        let malloc_id = self.module
            .declare_function("bolt_malloc", Linkage::Import, &malloc_sig)
            .map_err(|e| BoltError::Codegen { message: format!("Failed to declare bolt_malloc: {}", e) })?;
        self.func_names.insert("bolt_malloc".to_string(), malloc_id);

        // bolt_free(ptr: i64, size: i64)
        let mut free_sig = self.module.make_signature();
        free_sig.params.push(AbiParam::new(types::I64));
        free_sig.params.push(AbiParam::new(types::I64));
        let free_id = self.module
            .declare_function("bolt_free", Linkage::Import, &free_sig)
            .map_err(|e| BoltError::Codegen { message: format!("Failed to declare bolt_free: {}", e) })?;
        self.func_names.insert("bolt_free".to_string(), free_id);

        // bolt_realloc(ptr: i64, old_size: i64, new_size: i64) -> i64
        let mut realloc_sig = self.module.make_signature();
        realloc_sig.params.push(AbiParam::new(types::I64));
        realloc_sig.params.push(AbiParam::new(types::I64));
        realloc_sig.params.push(AbiParam::new(types::I64));
        realloc_sig.returns.push(AbiParam::new(types::I64));
        let realloc_id = self.module
            .declare_function("bolt_realloc", Linkage::Import, &realloc_sig)
            .map_err(|e| BoltError::Codegen { message: format!("Failed to declare bolt_realloc: {}", e) })?;
        self.func_names.insert("bolt_realloc".to_string(), realloc_id);

        // bolt_print_int(value: i64)
        let mut print_int_sig = self.module.make_signature();
        print_int_sig.params.push(AbiParam::new(types::I64));
        let print_int_id = self.module
            .declare_function("bolt_print_int", Linkage::Import, &print_int_sig)
            .map_err(|e| BoltError::Codegen { message: format!("Failed to declare bolt_print_int: {}", e) })?;
        self.func_names.insert("bolt_print_int".to_string(), print_int_id);

        // bolt_print_str(ptr: i64, len: i64)
        let mut print_str_sig = self.module.make_signature();
        print_str_sig.params.push(AbiParam::new(types::I64));
        print_str_sig.params.push(AbiParam::new(types::I64));
        let print_str_id = self.module
            .declare_function("bolt_print_str", Linkage::Import, &print_str_sig)
            .map_err(|e| BoltError::Codegen { message: format!("Failed to declare bolt_print_str: {}", e) })?;
        self.func_names.insert("bolt_print_str".to_string(), print_str_id);

        Ok(())
    }

    fn register_trait(&mut self, name: &str, t: &hir::Trait) {
        // For now, just store an empty method list
        // Full trait method parsing would extract method signatures
        self.trait_defs.insert(name.to_string(), TraitDef {
            methods: Vec::new(),
        });
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

        // Build method map by scanning functions with Type_method naming convention
        let mut methods = HashMap::new();
        for (_, item) in &krate.items {
            if let ItemKind::Function(_) = &item.kind {
                // Check if this is a method for this type
                if item.name.starts_with(&format!("{}_", type_name)) {
                    let method_name = item.name.strip_prefix(&format!("{}_", type_name))
                        .unwrap_or(&item.name)
                        .to_string();
                    if let Some(&func_id) = self.func_ids.get(&item.id) {
                        methods.insert(method_name, MethodImpl {
                            func_id,
                            func_name: item.name.clone(),
                        });
                    }
                }
            }
        }

        // Add impl block to type's impl list
        self.type_impls
            .entry(type_name)
            .or_insert_with(Vec::new)
            .push(ImplBlock {
                trait_name,
                methods,
            });
    }

    fn type_to_name(&self, ty: &HirType) -> String {
        match &ty.kind {
            TypeKind::Path(path) => {
                path.segments.last().map(|s| s.ident.clone()).unwrap_or_default()
            }
            _ => String::new(),
        }
    }

    fn register_struct(&mut self, name: &str, s: &Struct) {
        let mut fields = IndexMap::new();
        let mut offset = 0usize;
        
        if let StructKind::Named(ref field_list) = s.kind {
            for field in field_list {
                fields.insert(field.name.clone(), offset);
                offset += 8;  // All fields are 8 bytes for now
            }
        }
        
        self.struct_layouts.insert(name.to_string(), StructLayout { fields, size: offset });
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

        // If function returns a struct, use sret convention:
        // Add hidden first parameter for caller-allocated return space
        let returns_struct = self.is_struct_type(&func.sig.output);
        if returns_struct {
            sig.params.push(AbiParam::new(types::I64)); // sret pointer
        }

        for (_, ty) in &func.sig.inputs {
            sig.params.push(AbiParam::new(self.type_to_cl(ty)));
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
        let returns_struct = self.is_struct_type(&func.sig.output);

        self.ctx.func.signature = self.module.declarations().get_function_decl(func_id).signature.clone();

        // Pre-compute Cranelift types for parameters before creating builder
        let param_types: Vec<types::Type> = func.sig.inputs.iter()
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

        // Get block params before creating translator (offset by 1 if sret)
        let param_offset = if returns_struct { 1 } else { 0 };
        let param_values: Vec<_> = (0..func.sig.inputs.len())
            .map(|i| builder.block_params(entry_block)[i + param_offset])
            .collect();

        let mut translator = FunctionTranslator::new(builder, &self.type_registry, &self.func_ids, &self.func_names, &mut self.module, &self.struct_layouts, &self.enum_layouts, &self.type_impls, &self.imports);
        translator.sret_ptr = sret_ptr;

        for (((param_name, _), val), cl_ty) in func.sig.inputs.iter().zip(param_values).zip(param_types) {
            let var = Variable::from_u32(translator.next_var as u32);
            translator.next_var += 1;
            translator.builder.declare_var(var, cl_ty);
            translator.builder.def_var(var, val);
            translator.locals.insert(param_name.clone(), LocalInfo { var, ty: cl_ty });
        }

        if let Some(ref body) = func.body {
            let result = translator.translate_block(body);
            if let Some(val) = result {
                // If we have sret and returning a struct pointer, copy to sret location
                if let Some(sret) = translator.sret_ptr {
                    // The result is a pointer to a local struct - copy its contents to sret
                    // For now, just return sret as the result
                    translator.builder.ins().return_(&[sret]);
                } else {
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
            .map_err(|e| BoltError::Codegen { message: format!("Failed to define {}: {}", name, e) })?;

        self.module.clear_context(&mut self.ctx);
        Ok(())
    }

    /// Check if a type is a struct (returns by pointer)
    fn is_struct_type(&self, ty: &HirType) -> bool {
        if let TypeKind::Path(path) = &ty.kind {
            if let Some(seg) = path.segments.first() {
                return self.struct_layouts.contains_key(&seg.ident);
            }
        }
        false
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
    locals: IndexMap<String, LocalInfo>,
    local_types: HashMap<String, String>,  // variable name -> type name
    closure_defs: HashMap<String, ClosureDef>,
    next_var: usize,
    loop_stack: Vec<(Block, Block)>,
    sret_ptr: Option<Value>,  // pointer to caller-allocated return space for struct returns
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
            locals: IndexMap::new(),
            local_types: HashMap::new(),
            closure_defs: HashMap::new(),
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
                    self.local_types.get(name).cloned()
                } else {
                    None
                }
            }
            ExprKind::Struct { path, .. } => {
                // Struct literal - type is the struct name
                path.segments.first().map(|s| s.ident.clone())
            }
            ExprKind::Call { func, .. } => {
                // Function call - check if it's a constructor (Type::new pattern)
                if let ExprKind::Path(path) = &func.kind {
                    if path.segments.len() == 2 {
                        // Type::method() - return the type name
                        return Some(path.segments[0].ident.clone());
                    }
                }
                None
            }
            ExprKind::MethodCall { receiver, .. } => {
                // Recursive - infer from receiver
                self.infer_type_name(receiver)
            }
            ExprKind::Field { expr: inner, .. } => {
                // Field access - infer from inner
                self.infer_type_name(inner)
            }
            _ => None,
        }
    }

    /// Resolve a method for a given type
    fn resolve_method(&self, type_name: &str, method_name: &str) -> Option<FuncId> {
        // First check type_impls registry
        if let Some(impls) = self.type_impls.get(type_name) {
            for impl_block in impls {
                if let Some(method) = impl_block.methods.get(method_name) {
                    return Some(method.func_id);
                }
            }
        }

        // Fallback to naming convention: Type_method
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
                                path.segments.first().map(|s| s.ident.clone())
                            } else {
                                None
                            }
                        })
                    });

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
                            }
                        }
                    }
                    // Handle pointer dereference assignment: *ptr = value
                    ExprKind::Deref(inner) => {
                        let ptr = self.translate_expr(inner);
                        self.builder.ins().store(cranelift::prelude::MemFlags::new(), rhs_val, ptr, 0);
                    }
                    // Handle indexed assignment: arr[i] = value
                    ExprKind::Index { expr: base, index } => {
                        let ptr = self.translate_expr(base);
                        let idx = self.translate_expr(index);
                        let eight = self.builder.ins().iconst(types::I64, 8);
                        let offset = self.builder.ins().imul(idx, eight);
                        let addr = self.builder.ins().iadd(ptr, offset);
                        self.builder.ins().store(cranelift::prelude::MemFlags::new(), rhs_val, addr, 0);
                    }
                    // Handle field assignment: s.field = value
                    ExprKind::Field { expr: base, field } => {
                        let ptr = self.translate_expr(base);
                        for layout in self.struct_layouts.values() {
                            if let Some(&offset) = layout.fields.get(field) {
                                self.builder.ins().store(
                                    cranelift::prelude::MemFlags::new(),
                                    rhs_val,
                                    ptr,
                                    offset as i32,
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
                        if let Some(&func_id) = self.func_names.get(&full_path) {
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

                            // Add regular arguments
                            arg_vals.extend(args.iter().map(|arg| self.translate_expr(arg)));

                            let call = self.builder.ins().call(func_ref, &arg_vals);
                            let results = self.builder.inst_results(call);
                            if !results.is_empty() {
                                return results[0];
                            }
                        }
                    }
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident;

                        // First check if this name is an import alias
                        let resolved_name = self.imports.get(name).cloned().unwrap_or_else(|| name.clone());

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

                            // Add regular arguments
                            arg_vals.extend(args.iter().map(|arg| self.translate_expr(arg)));

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

                // Try to resolve the method using the type registry
                if let Some(ref tn) = type_name {
                    if let Some(func_id) = self.resolve_method(tn, method) {
                        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);

                        // Build args: self first, then rest
                        let mut arg_vals = vec![recv_val];
                        arg_vals.extend(args.iter().map(|arg| self.translate_expr(arg)));

                        let call = self.builder.ins().call(func_ref, &arg_vals);
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

                        // Build args: self first, then rest
                        let mut arg_vals = vec![recv_val];
                        arg_vals.extend(args.iter().map(|arg| self.translate_expr(arg)));

                        let call = self.builder.ins().call(func_ref, &arg_vals);
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
                    self.translate_expr(&elems[0])
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
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Continue { .. } => {
                if let Some(&(header_block, _)) = self.loop_stack.last() {
                    self.builder.ins().jump(header_block, &[]);
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Struct { path, fields, .. } => {
                // Get struct name from path
                if let Some(seg) = path.segments.first() {
                    let struct_name = &seg.ident;
                    if let Some(layout) = self.struct_layouts.get(struct_name) {
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
                            if let Some(&offset) = layout.fields.get(field_name) {
                                let val = self.translate_expr(field_expr);
                                self.builder.ins().store(
                                    cranelift::prelude::MemFlags::new(),
                                    val,
                                    ptr,
                                    offset as i32,
                                );
                            }
                        }

                        return ptr;
                    }
                }
                self.builder.ins().iconst(types::I64, 0)
            }
            ExprKind::Field { expr: base, field } => {
                let ptr = self.translate_expr(base);
                // Try to find field offset - for now we'll check all known structs
                for layout in self.struct_layouts.values() {
                    if let Some(&offset) = layout.fields.get(field) {
                        return self.builder.ins().load(
                            types::I64,
                            cranelift::prelude::MemFlags::new(),
                            ptr,
                            offset as i32,
                        );
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
                let ptr = self.translate_expr(base);
                let idx = self.translate_expr(index);
                // Calculate offset: idx * 8
                let eight = self.builder.ins().iconst(types::I64, 8);
                let offset = self.builder.ins().imul(idx, eight);
                let addr = self.builder.ins().iadd(ptr, offset);
                self.builder.ins().load(types::I64, cranelift::prelude::MemFlags::new(), addr, 0)
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
                // For variables, we need to spill to stack and return address
                if let ExprKind::Path(path) = &inner.kind {
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident;
                        if let Some(info) = self.locals.get(name) {
                            let val = self.builder.use_var(info.var);
                            // Allocate stack slot and store value
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

    fn translate_binop(&mut self, op: BinaryOp, lhs: Value, rhs: Value) -> Value {
        // Check if operands are floats by inspecting the value type
        let lhs_ty = self.builder.func.dfg.value_type(lhs);
        let is_float = lhs_ty == types::F32 || lhs_ty == types::F64;

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
