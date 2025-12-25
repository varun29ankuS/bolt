//! Unified Type System for Bolt
//!
//! This module provides a single source of truth for types throughout compilation.
//! It handles:
//! - Type definitions (structs, enums, type aliases)
//! - Type resolution (resolving Path types to concrete definitions)
//! - Monomorphization (specializing generic types/functions)
//! - Type layout calculation for codegen

use crate::error::Span;
use crate::hir::{
    self, Crate, DefId, Enum, FloatType, GenericParam, Generics, IntType, ItemKind, Path,
    PathSegment, Struct, StructKind, Type as HirType, TypeKind, UintType,
};
use indexmap::IndexMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// A unique identifier for a resolved type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyId(pub u32);

/// A unique identifier for a lifetime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LifetimeId(pub u32);

/// Represents a lifetime in the type system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Lifetime {
    /// The static lifetime 'static
    Static,
    /// A named lifetime parameter (e.g., 'a)
    Named(String),
    /// An inferred lifetime (assigned during type checking)
    Infer(u32),
    /// An anonymous/elided lifetime
    Anonymous,
    /// Lifetime bound to a specific scope
    Bound {
        /// Scope depth (higher = inner scope)
        scope_depth: u32,
        /// Index within the scope
        index: u32,
    },
}

impl Lifetime {
    pub fn is_static(&self) -> bool {
        matches!(self, Lifetime::Static)
    }

    pub fn is_infer(&self) -> bool {
        matches!(self, Lifetime::Infer(_))
    }
}

/// The resolved, canonical form of a type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ty {
    /// Unit type `()`
    Unit,
    /// Boolean
    Bool,
    /// Character
    Char,
    /// Signed integer
    Int(IntType),
    /// Unsigned integer
    Uint(UintType),
    /// Floating point
    Float(FloatType),
    /// String slice `str`
    Str,
    /// Reference `&T` or `&mut T`
    Ref {
        lifetime: LifetimeId,
        mutable: bool,
        inner: TyId,
    },
    /// Raw pointer `*const T` or `*mut T`
    Ptr {
        mutable: bool,
        inner: TyId,
    },
    /// Slice `[T]`
    Slice(TyId),
    /// Array `[T; N]`
    Array {
        elem: TyId,
        len: usize,
    },
    /// Tuple `(T, U, V)`
    Tuple(Vec<TyId>),
    /// Function type `fn(A, B) -> C`
    Fn {
        inputs: Vec<TyId>,
        output: TyId,
    },
    /// A struct definition with its concrete type arguments
    Adt {
        def_id: DefId,
        name: String,
        /// Substituted type arguments (empty if non-generic)
        args: Vec<TyId>,
    },
    /// Never type `!`
    Never,
    /// Type inference variable
    Infer(u32),
    /// Type parameter (generic) - resolved during monomorphization
    Param {
        name: String,
        index: usize,
    },
    /// Error type for recovery
    Error,

    // ===== Built-in collection types =====

    /// Vec<T> - growable array with heap allocation
    /// Layout: ptr (8) + len (8) + capacity (8) = 24 bytes
    Vec(TyId),

    /// String - owned UTF-8 string (essentially Vec<u8>)
    /// Layout: ptr (8) + len (8) + capacity (8) = 24 bytes
    String,

    /// Box<T> - heap-allocated value
    /// Layout: ptr (8) = 8 bytes
    Box(TyId),

    /// Option<T> - optional value
    /// Layout: discriminant (1) + padding + T
    Option(TyId),

    /// Result<T, E> - result with error
    /// Layout: discriminant (1) + padding + max(T, E)
    Result { ok: TyId, err: TyId },

    /// impl Trait - existential type that implements trait bounds
    /// At monomorphization, this resolves to the concrete type
    ImplTrait {
        /// Trait bounds as trait names
        bounds: Vec<String>,
        /// The concrete type this resolves to (filled in during inference)
        concrete: Option<TyId>,
    },

    /// dyn Trait - trait object (fat pointer)
    /// Layout: ptr (8) + vtable ptr (8) = 16 bytes
    DynTrait {
        /// Trait bounds as trait names
        bounds: Vec<String>,
    },
}

/// Layout information for a type
#[derive(Debug, Clone)]
pub struct TypeLayout {
    /// Size in bytes
    pub size: usize,
    /// Alignment in bytes
    pub align: usize,
    /// For structs: field offsets
    pub fields: Option<IndexMap<String, FieldLayout>>,
    /// For enums: variant layouts
    pub variants: Option<IndexMap<String, VariantLayout>>,
}

#[derive(Debug, Clone)]
pub struct FieldLayout {
    pub offset: usize,
    pub ty: TyId,
}

#[derive(Debug, Clone)]
pub struct VariantLayout {
    pub discriminant: i64,
    pub fields: Option<IndexMap<String, FieldLayout>>,
    pub payload_size: usize,
}

/// Definition of a struct
#[derive(Debug, Clone)]
pub struct StructDef {
    pub def_id: DefId,
    pub name: String,
    pub generics: Generics,
    pub fields: Vec<(String, HirType)>,
    pub is_tuple: bool,
}

/// Definition of an enum
#[derive(Debug, Clone)]
pub struct EnumDef {
    pub def_id: DefId,
    pub name: String,
    pub generics: Generics,
    pub variants: Vec<VariantDef>,
}

#[derive(Debug, Clone)]
pub struct VariantDef {
    pub name: String,
    pub fields: Vec<(String, HirType)>,
    pub is_tuple: bool,
}

/// A monomorphized function instance
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MonoKey {
    pub base_name: String,
    pub type_args: Vec<TyId>,
}

// ============================================================================
// Method Resolution System
// ============================================================================

/// Information about a resolved method
#[derive(Debug, Clone)]
pub struct MethodInfo {
    /// The DefId of the method's function definition
    pub def_id: DefId,
    /// The mangled name used in codegen (e.g., "Vec_push")
    pub mangled_name: String,
    /// Whether this is a trait method (vs inherent impl)
    pub is_trait_method: bool,
    /// The trait this method belongs to (if any)
    pub trait_name: Option<String>,
    /// Number of type parameters on the method itself
    pub method_type_params: usize,
}

/// Method lookup table for fast resolution
#[derive(Debug, Default)]
pub struct MethodTable {
    /// Inherent methods: (TypeName, MethodName) -> MethodInfo
    inherent_methods: HashMap<(String, String), MethodInfo>,
    /// Trait methods: (TraitName, MethodName) -> DefId of trait method
    trait_methods: HashMap<(String, String), DefId>,
    /// Which traits each type implements: TypeName -> Vec<TraitName>
    trait_impls: HashMap<String, Vec<String>>,
    /// Trait impl methods: (TypeName, TraitName, MethodName) -> MethodInfo
    trait_impl_methods: HashMap<(String, String, String), MethodInfo>,
    /// Deref targets: TypeName -> TargetTypeName (for deref coercion)
    deref_targets: HashMap<String, String>,
}

impl MethodTable {
    pub fn new() -> Self {
        let mut table = Self::default();
        // Register known deref chains
        table.deref_targets.insert("String".to_string(), "str".to_string());
        table.deref_targets.insert("Box".to_string(), "__inner__".to_string()); // placeholder
        table.deref_targets.insert("Vec".to_string(), "__slice__".to_string()); // placeholder
        table
    }

    /// Register an inherent method (impl Type { fn method() })
    pub fn register_inherent_method(
        &mut self,
        type_name: &str,
        method_name: &str,
        def_id: DefId,
        method_type_params: usize,
    ) {
        let mangled = format!("{}_{}", type_name, method_name);
        self.inherent_methods.insert(
            (type_name.to_string(), method_name.to_string()),
            MethodInfo {
                def_id,
                mangled_name: mangled,
                is_trait_method: false,
                trait_name: None,
                method_type_params,
            },
        );
    }

    /// Register a trait method definition
    pub fn register_trait_method(&mut self, trait_name: &str, method_name: &str, def_id: DefId) {
        self.trait_methods.insert(
            (trait_name.to_string(), method_name.to_string()),
            def_id,
        );
    }

    /// Register that a type implements a trait
    pub fn register_trait_impl(&mut self, type_name: &str, trait_name: &str) {
        self.trait_impls
            .entry(type_name.to_string())
            .or_default()
            .push(trait_name.to_string());
    }

    /// Register a trait impl method (impl Trait for Type { fn method() })
    pub fn register_trait_impl_method(
        &mut self,
        type_name: &str,
        trait_name: &str,
        method_name: &str,
        def_id: DefId,
    ) {
        let mangled = format!("{}_{}_{}", type_name, trait_name, method_name);
        self.trait_impl_methods.insert(
            (type_name.to_string(), trait_name.to_string(), method_name.to_string()),
            MethodInfo {
                def_id,
                mangled_name: mangled,
                is_trait_method: true,
                trait_name: Some(trait_name.to_string()),
                method_type_params: 0,
            },
        );
    }

    /// Look up an inherent method
    pub fn get_inherent_method(&self, type_name: &str, method_name: &str) -> Option<&MethodInfo> {
        self.inherent_methods.get(&(type_name.to_string(), method_name.to_string()))
    }

    /// Look up a trait impl method
    pub fn get_trait_impl_method(
        &self,
        type_name: &str,
        trait_name: &str,
        method_name: &str,
    ) -> Option<&MethodInfo> {
        self.trait_impl_methods.get(&(
            type_name.to_string(),
            trait_name.to_string(),
            method_name.to_string(),
        ))
    }

    /// Get all traits implemented by a type
    pub fn get_trait_impls(&self, type_name: &str) -> &[String] {
        self.trait_impls
            .get(type_name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get deref target for a type
    pub fn get_deref_target(&self, type_name: &str) -> Option<&str> {
        self.deref_targets.get(type_name).map(|s| s.as_str())
    }
}

/// A lifetime constraint: lhs outlives rhs ('lhs: 'rhs)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LifetimeConstraint {
    /// The lifetime that must live longer
    pub lhs: LifetimeId,
    /// The lifetime that must be outlived
    pub rhs: LifetimeId,
    /// Source span for error reporting
    pub span: Option<Span>,
}

/// The type registry - single source of truth for all types
pub struct TypeRegistry {
    /// All resolved types, indexed by TyId
    types: RwLock<Vec<Ty>>,
    /// Interning map: Ty -> TyId for deduplication
    intern_map: RwLock<HashMap<Ty, TyId>>,
    /// Next type ID counter
    next_id: AtomicU32,
    /// Next inference variable ID
    next_infer: AtomicU32,
    /// All lifetimes, indexed by LifetimeId
    lifetimes: RwLock<Vec<Lifetime>>,
    /// Lifetime interning map
    lifetime_intern_map: RwLock<HashMap<Lifetime, LifetimeId>>,
    /// Next lifetime ID counter
    next_lifetime_id: AtomicU32,
    /// Next lifetime inference variable ID
    next_lifetime_infer: AtomicU32,
    /// Lifetime constraints collected during type checking
    lifetime_constraints: RwLock<Vec<LifetimeConstraint>>,
    /// Struct definitions by name
    struct_defs: RwLock<HashMap<String, StructDef>>,
    /// Enum definitions by name
    enum_defs: RwLock<HashMap<String, EnumDef>>,
    /// Type alias definitions by name
    type_aliases: RwLock<HashMap<String, (Generics, HirType)>>,
    /// Cached layouts
    layouts: RwLock<HashMap<TyId, TypeLayout>>,
    /// Monomorphized function instances
    mono_instances: RwLock<HashMap<MonoKey, String>>,
    /// Expression types by span - populated during type checking
    expr_types: RwLock<HashMap<Span, TyId>>,
    /// Local variable types by (function_name, var_name)
    local_types: RwLock<HashMap<(String, String), TyId>>,
    /// Method resolution table
    method_table: RwLock<MethodTable>,
}

impl TypeRegistry {
    pub fn new() -> Self {
        let registry = Self {
            types: RwLock::new(Vec::new()),
            intern_map: RwLock::new(HashMap::new()),
            next_id: AtomicU32::new(0),
            next_infer: AtomicU32::new(0),
            lifetimes: RwLock::new(Vec::new()),
            lifetime_intern_map: RwLock::new(HashMap::new()),
            next_lifetime_id: AtomicU32::new(0),
            next_lifetime_infer: AtomicU32::new(0),
            lifetime_constraints: RwLock::new(Vec::new()),
            struct_defs: RwLock::new(HashMap::new()),
            enum_defs: RwLock::new(HashMap::new()),
            type_aliases: RwLock::new(HashMap::new()),
            layouts: RwLock::new(HashMap::new()),
            mono_instances: RwLock::new(HashMap::new()),
            expr_types: RwLock::new(HashMap::new()),
            local_types: RwLock::new(HashMap::new()),
            method_table: RwLock::new(MethodTable::new()),
        };
        // Pre-intern the static lifetime
        registry.intern_lifetime(Lifetime::Static);
        registry
    }

    /// Intern a lifetime, returning its unique ID
    pub fn intern_lifetime(&self, lifetime: Lifetime) -> LifetimeId {
        // Check if already interned
        if let Some(&id) = self.lifetime_intern_map.read().get(&lifetime) {
            return id;
        }

        // Allocate new ID
        let id = LifetimeId(self.next_lifetime_id.fetch_add(1, Ordering::Relaxed));
        self.lifetimes.write().push(lifetime.clone());
        self.lifetime_intern_map.write().insert(lifetime, id);
        id
    }

    /// Get a lifetime by its ID
    pub fn get_lifetime(&self, id: LifetimeId) -> Option<Lifetime> {
        self.lifetimes.read().get(id.0 as usize).cloned()
    }

    /// Get the static lifetime ID (always ID 0)
    pub fn static_lifetime(&self) -> LifetimeId {
        LifetimeId(0)
    }

    /// Create a fresh lifetime inference variable
    pub fn fresh_lifetime_infer(&self) -> LifetimeId {
        let infer_id = self.next_lifetime_infer.fetch_add(1, Ordering::Relaxed);
        self.intern_lifetime(Lifetime::Infer(infer_id))
    }

    /// Create an anonymous/elided lifetime
    pub fn anonymous_lifetime(&self) -> LifetimeId {
        self.intern_lifetime(Lifetime::Anonymous)
    }

    /// Create a named lifetime
    pub fn named_lifetime(&self, name: &str) -> LifetimeId {
        self.intern_lifetime(Lifetime::Named(name.to_string()))
    }

    /// Add a lifetime constraint: lhs must outlive rhs
    pub fn add_lifetime_constraint(&self, lhs: LifetimeId, rhs: LifetimeId, span: Option<Span>) {
        self.lifetime_constraints.write().push(LifetimeConstraint { lhs, rhs, span });
    }

    /// Get all lifetime constraints
    pub fn get_lifetime_constraints(&self) -> Vec<LifetimeConstraint> {
        self.lifetime_constraints.read().clone()
    }

    /// Clear lifetime constraints (e.g., after checking a function)
    pub fn clear_lifetime_constraints(&self) {
        self.lifetime_constraints.write().clear();
    }

    /// Initialize the registry from a crate's type definitions
    pub fn init_from_crate(&self, krate: &Crate) {
        // First pass: register type definitions
        for (def_id, item) in &krate.items {
            match &item.kind {
                ItemKind::Struct(s) => {
                    self.register_struct(*def_id, &item.name, s);
                }
                ItemKind::Enum(e) => {
                    self.register_enum(*def_id, &item.name, e);
                }
                ItemKind::TypeAlias(ta) => {
                    self.register_type_alias(&item.name, &ta.generics, &ta.ty);
                }
                ItemKind::Trait(t) => {
                    self.register_trait(&item.name, t, krate);
                }
                _ => {}
            }
        }

        // Second pass: register impl blocks and methods
        for (def_id, item) in &krate.items {
            if let ItemKind::Impl(impl_block) = &item.kind {
                self.register_impl(*def_id, impl_block, krate);
            }
        }

        // Third pass: register standalone methods (Type_method pattern)
        for (def_id, item) in &krate.items {
            if let ItemKind::Function(f) = &item.kind {
                // Check for mangled method names like "Vec_push", "String_len"
                if let Some(underscore_pos) = item.name.find('_') {
                    let type_name = &item.name[..underscore_pos];
                    let method_name = &item.name[underscore_pos + 1..];

                    // Only register if this looks like a method (has type prefix)
                    if self.struct_defs.read().contains_key(type_name)
                        || self.enum_defs.read().contains_key(type_name)
                        || ["Vec", "String", "Box", "Option", "Result"].contains(&type_name)
                    {
                        let type_params = f.sig.generics.params.iter()
                            .filter(|p| matches!(p, GenericParam::Type { .. }))
                            .count();
                        self.method_table.write().register_inherent_method(
                            type_name,
                            method_name,
                            *def_id,
                            type_params,
                        );
                    }
                }
            }
        }
    }

    fn register_trait(&self, name: &str, t: &hir::Trait, krate: &Crate) {
        // Register trait method signatures
        for &method_def_id in &t.items {
            if let Some(method_item) = krate.items.get(&method_def_id) {
                if let ItemKind::Function(_) = &method_item.kind {
                    self.method_table.write().register_trait_method(
                        name,
                        &method_item.name,
                        method_def_id,
                    );
                }
            }
        }
    }

    fn register_impl(&self, _def_id: DefId, impl_block: &hir::Impl, krate: &Crate) {
        // Get the type name this impl is for
        let type_name = self.extract_type_name(&impl_block.self_ty);

        // Check if this is a trait impl or inherent impl
        if let Some(trait_ref) = &impl_block.trait_ref {
            // Trait impl: impl Trait for Type
            let trait_name = trait_ref.segments.last()
                .map(|s| s.ident.as_str())
                .unwrap_or("");

            self.method_table.write().register_trait_impl(&type_name, trait_name);

            // Register each method
            for &method_def_id in &impl_block.items {
                if let Some(method_item) = krate.items.get(&method_def_id) {
                    if let ItemKind::Function(_) = &method_item.kind {
                        // Extract just the method name (remove type prefix if present)
                        let method_name = method_item.name
                            .split('_')
                            .last()
                            .unwrap_or(&method_item.name);

                        self.method_table.write().register_trait_impl_method(
                            &type_name,
                            trait_name,
                            method_name,
                            method_def_id,
                        );
                    }
                }
            }
        } else {
            // Inherent impl: impl Type
            for &method_def_id in &impl_block.items {
                if let Some(method_item) = krate.items.get(&method_def_id) {
                    if let ItemKind::Function(f) = &method_item.kind {
                        // Extract just the method name
                        let method_name = method_item.name
                            .split('_')
                            .last()
                            .unwrap_or(&method_item.name);

                        let type_params = f.sig.generics.params.iter()
                            .filter(|p| matches!(p, GenericParam::Type { .. }))
                            .count();

                        self.method_table.write().register_inherent_method(
                            &type_name,
                            method_name,
                            method_def_id,
                            type_params,
                        );
                    }
                }
            }
        }
    }

    fn extract_type_name(&self, ty: &HirType) -> String {
        match &ty.kind {
            TypeKind::Path(path) => {
                path.segments.last()
                    .map(|s| s.ident.clone())
                    .unwrap_or_else(|| "Unknown".to_string())
            }
            _ => "Unknown".to_string(),
        }
    }

    fn register_struct(&self, def_id: DefId, name: &str, s: &Struct) {
        let fields = match &s.kind {
            StructKind::Unit => vec![],
            StructKind::Tuple(types) => types
                .iter()
                .enumerate()
                .map(|(i, ty)| (i.to_string(), ty.clone()))
                .collect(),
            StructKind::Named(fields) => fields
                .iter()
                .map(|f| (f.name.clone(), f.ty.clone()))
                .collect(),
        };

        self.struct_defs.write().insert(
            name.to_string(),
            StructDef {
                def_id,
                name: name.to_string(),
                generics: s.generics.clone(),
                fields,
                is_tuple: matches!(s.kind, StructKind::Tuple(_)),
            },
        );
    }

    fn register_enum(&self, def_id: DefId, name: &str, e: &Enum) {
        let variants = e
            .variants
            .iter()
            .map(|v| {
                let (fields, is_tuple) = match &v.kind {
                    StructKind::Unit => (vec![], false),
                    StructKind::Tuple(types) => (
                        types
                            .iter()
                            .enumerate()
                            .map(|(i, ty)| (i.to_string(), ty.clone()))
                            .collect(),
                        true,
                    ),
                    StructKind::Named(fields) => (
                        fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect(),
                        false,
                    ),
                };
                VariantDef {
                    name: v.name.clone(),
                    fields,
                    is_tuple,
                }
            })
            .collect();

        self.enum_defs.write().insert(
            name.to_string(),
            EnumDef {
                def_id,
                name: name.to_string(),
                generics: e.generics.clone(),
                variants,
            },
        );
    }

    fn register_type_alias(&self, name: &str, generics: &Generics, ty: &HirType) {
        self.type_aliases
            .write()
            .insert(name.to_string(), (generics.clone(), ty.clone()));
    }

    /// Intern a type, returning its TyId (reuses existing if identical)
    pub fn intern(&self, ty: Ty) -> TyId {
        // Check if we already have this type
        if let Some(&id) = self.intern_map.read().get(&ty) {
            return id;
        }

        // Create new type
        let id = TyId(self.next_id.fetch_add(1, Ordering::SeqCst));
        self.types.write().push(ty.clone());
        self.intern_map.write().insert(ty, id);
        id
    }

    /// Get a type by its ID
    pub fn get(&self, id: TyId) -> Option<Ty> {
        self.types.read().get(id.0 as usize).cloned()
    }

    /// Create a fresh inference variable
    pub fn fresh_infer(&self) -> TyId {
        let var = self.next_infer.fetch_add(1, Ordering::SeqCst);
        self.intern(Ty::Infer(var))
    }

    /// Resolve a HIR type to a TyId
    pub fn resolve_hir_type(&self, ty: &HirType) -> TyId {
        self.resolve_hir_type_with_subst(ty, &HashMap::new())
    }

    /// Resolve a HIR type with generic substitutions
    pub fn resolve_hir_type_with_subst(
        &self,
        ty: &HirType,
        subst: &HashMap<String, TyId>,
    ) -> TyId {
        match &ty.kind {
            TypeKind::Unit => self.intern(Ty::Unit),
            TypeKind::Bool => self.intern(Ty::Bool),
            TypeKind::Char => self.intern(Ty::Char),
            TypeKind::Int(i) => self.intern(Ty::Int(*i)),
            TypeKind::Uint(u) => self.intern(Ty::Uint(*u)),
            TypeKind::Float(f) => self.intern(Ty::Float(*f)),
            TypeKind::Str => self.intern(Ty::Str),
            TypeKind::Never => self.intern(Ty::Never),
            TypeKind::Infer => self.fresh_infer(),
            TypeKind::Error => self.intern(Ty::Error),

            TypeKind::Ref { lifetime, mutable, inner } => {
                let inner_id = self.resolve_hir_type_with_subst(inner, subst);
                // Convert HIR lifetime (Option<String>) to LifetimeId
                let lifetime_id = match lifetime {
                    Some(name) if name == "static" => self.static_lifetime(),
                    Some(name) => self.named_lifetime(name),
                    None => self.anonymous_lifetime(),
                };
                self.intern(Ty::Ref {
                    lifetime: lifetime_id,
                    mutable: *mutable,
                    inner: inner_id,
                })
            }

            TypeKind::Ptr { mutable, inner } => {
                let inner_id = self.resolve_hir_type_with_subst(inner, subst);
                self.intern(Ty::Ptr {
                    mutable: *mutable,
                    inner: inner_id,
                })
            }

            TypeKind::Slice(inner) => {
                let inner_id = self.resolve_hir_type_with_subst(inner, subst);
                self.intern(Ty::Slice(inner_id))
            }

            TypeKind::Array { elem, len } => {
                let elem_id = self.resolve_hir_type_with_subst(elem, subst);
                self.intern(Ty::Array { elem: elem_id, len: *len })
            }

            TypeKind::Tuple(elems) => {
                let elem_ids: Vec<_> = elems
                    .iter()
                    .map(|e| self.resolve_hir_type_with_subst(e, subst))
                    .collect();
                self.intern(Ty::Tuple(elem_ids))
            }

            TypeKind::Fn { inputs, output } => {
                let input_ids: Vec<_> = inputs
                    .iter()
                    .map(|i| self.resolve_hir_type_with_subst(i, subst))
                    .collect();
                let output_id = self.resolve_hir_type_with_subst(output, subst);
                self.intern(Ty::Fn {
                    inputs: input_ids,
                    output: output_id,
                })
            }

            TypeKind::Path(path) => self.resolve_path_type(path, subst),

            TypeKind::ImplTrait(bounds) => {
                let trait_names = self.resolve_type_bounds(bounds);
                self.intern(Ty::ImplTrait {
                    bounds: trait_names,
                    concrete: None,
                })
            }

            TypeKind::DynTrait(bounds) => {
                let trait_names = self.resolve_type_bounds(bounds);
                self.intern(Ty::DynTrait {
                    bounds: trait_names,
                })
            }
        }
    }

    /// Resolve HIR type bounds to trait names
    fn resolve_type_bounds(&self, bounds: &[hir::TypeBound]) -> Vec<String> {
        bounds
            .iter()
            .filter_map(|bound| {
                match bound {
                    hir::TypeBound::Trait(path) => {
                        // Convert path to trait name string
                        if !path.segments.is_empty() {
                            Some(path.segments.iter()
                                .map(|s| s.ident.clone())
                                .collect::<Vec<_>>()
                                .join("::"))
                        } else {
                            None
                        }
                    }
                    hir::TypeBound::Lifetime(_) => None, // Skip lifetime bounds for now
                }
            })
            .collect()
    }

    /// Resolve a path type (e.g., `Vec<i32>`, `String`, `MyStruct`)
    fn resolve_path_type(&self, path: &Path, subst: &HashMap<String, TyId>) -> TyId {
        if path.segments.is_empty() {
            return self.intern(Ty::Error);
        }

        // Single segment - could be a type parameter, primitive, or user type
        if path.segments.len() == 1 {
            let segment = &path.segments[0];
            let name = &segment.ident;

            // Check if it's a substituted type parameter
            if let Some(&ty_id) = subst.get(name) {
                return ty_id;
            }

            // Check for built-in primitive types
            match name.as_str() {
                "bool" => return self.intern(Ty::Bool),
                "char" => return self.intern(Ty::Char),
                "str" => return self.intern(Ty::Str),
                "i8" => return self.intern(Ty::Int(IntType::I8)),
                "i16" => return self.intern(Ty::Int(IntType::I16)),
                "i32" => return self.intern(Ty::Int(IntType::I32)),
                "i64" => return self.intern(Ty::Int(IntType::I64)),
                "i128" => return self.intern(Ty::Int(IntType::I128)),
                "isize" => return self.intern(Ty::Int(IntType::Isize)),
                "u8" => return self.intern(Ty::Uint(UintType::U8)),
                "u16" => return self.intern(Ty::Uint(UintType::U16)),
                "u32" => return self.intern(Ty::Uint(UintType::U32)),
                "u64" => return self.intern(Ty::Uint(UintType::U64)),
                "u128" => return self.intern(Ty::Uint(UintType::U128)),
                "usize" => return self.intern(Ty::Uint(UintType::Usize)),
                "f32" => return self.intern(Ty::Float(FloatType::F32)),
                "f64" => return self.intern(Ty::Float(FloatType::F64)),
                _ => {}
            }

            // Resolve type arguments if present
            let type_args = self.resolve_generic_args(segment, subst);

            // Check for built-in collection/wrapper types
            match name.as_str() {
                "Vec" => {
                    let elem_ty = type_args.first().copied().unwrap_or_else(|| self.intern(Ty::Error));
                    return self.intern(Ty::Vec(elem_ty));
                }
                "String" => {
                    return self.intern(Ty::String);
                }
                "Box" => {
                    let inner_ty = type_args.first().copied().unwrap_or_else(|| self.intern(Ty::Error));
                    return self.intern(Ty::Box(inner_ty));
                }
                "Option" => {
                    let inner_ty = type_args.first().copied().unwrap_or_else(|| self.intern(Ty::Error));
                    return self.intern(Ty::Option(inner_ty));
                }
                "Result" => {
                    let ok_ty = type_args.first().copied().unwrap_or_else(|| self.intern(Ty::Error));
                    let err_ty = type_args.get(1).copied().unwrap_or_else(|| self.intern(Ty::Error));
                    return self.intern(Ty::Result { ok: ok_ty, err: err_ty });
                }
                _ => {}
            }

            // Check for struct definition
            if let Some(struct_def) = self.struct_defs.read().get(name).cloned() {
                return self.intern(Ty::Adt {
                    def_id: struct_def.def_id,
                    name: name.clone(),
                    args: type_args,
                });
            }

            // Check for enum definition
            if let Some(enum_def) = self.enum_defs.read().get(name).cloned() {
                return self.intern(Ty::Adt {
                    def_id: enum_def.def_id,
                    name: name.clone(),
                    args: type_args,
                });
            }

            // Check for type alias
            if let Some((generics, target_ty)) = self.type_aliases.read().get(name).cloned() {
                // Build substitution map for the alias
                let mut alias_subst = subst.clone();
                for (i, param) in generics.params.iter().enumerate() {
                    if let GenericParam::Type { name: param_name, .. } = param {
                        if i < type_args.len() {
                            alias_subst.insert(param_name.clone(), type_args[i]);
                        }
                    }
                }
                return self.resolve_hir_type_with_subst(&target_ty, &alias_subst);
            }

            // Unknown type - might be a type parameter not in subst
            // Return as a Param type
            return self.intern(Ty::Param {
                name: name.clone(),
                index: 0,
            });
        }

        // Multi-segment path (e.g., `std::vec::Vec<T>`)
        // Use the last segment's name
        let last = path.segments.last().unwrap();
        let name = &last.ident;
        let type_args = self.resolve_generic_args(last, subst);

        // Try to resolve as struct or enum
        if let Some(struct_def) = self.struct_defs.read().get(name).cloned() {
            return self.intern(Ty::Adt {
                def_id: struct_def.def_id,
                name: name.clone(),
                args: type_args,
            });
        }

        if let Some(enum_def) = self.enum_defs.read().get(name).cloned() {
            return self.intern(Ty::Adt {
                def_id: enum_def.def_id,
                name: name.clone(),
                args: type_args,
            });
        }

        // Unknown path type
        self.intern(Ty::Error)
    }

    fn resolve_generic_args(
        &self,
        segment: &PathSegment,
        subst: &HashMap<String, TyId>,
    ) -> Vec<TyId> {
        segment
            .args
            .as_ref()
            .map(|args| {
                args.args
                    .iter()
                    .filter_map(|arg| match arg {
                        hir::GenericArg::Type(ty) => {
                            Some(self.resolve_hir_type_with_subst(ty, subst))
                        }
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get or create a monomorphized function name
    pub fn get_mono_name(&self, base_name: &str, type_args: &[TyId]) -> String {
        if type_args.is_empty() {
            return base_name.to_string();
        }

        let key = MonoKey {
            base_name: base_name.to_string(),
            type_args: type_args.to_vec(),
        };

        // Check if already exists
        if let Some(name) = self.mono_instances.read().get(&key) {
            return name.clone();
        }

        // Generate new name
        let type_suffix: String = type_args
            .iter()
            .map(|&id| self.mangle_type(id))
            .collect::<Vec<_>>()
            .join("_");

        let mono_name = format!("{}__{}", base_name, type_suffix);

        self.mono_instances.write().insert(key, mono_name.clone());
        mono_name
    }

    /// Mangle a type for use in function names
    fn mangle_type(&self, ty_id: TyId) -> String {
        match self.get(ty_id) {
            Some(Ty::Unit) => "unit".to_string(),
            Some(Ty::Bool) => "bool".to_string(),
            Some(Ty::Char) => "char".to_string(),
            Some(Ty::Int(IntType::I8)) => "i8".to_string(),
            Some(Ty::Int(IntType::I16)) => "i16".to_string(),
            Some(Ty::Int(IntType::I32)) => "i32".to_string(),
            Some(Ty::Int(IntType::I64)) => "i64".to_string(),
            Some(Ty::Int(IntType::I128)) => "i128".to_string(),
            Some(Ty::Int(IntType::Isize)) => "isize".to_string(),
            Some(Ty::Uint(UintType::U8)) => "u8".to_string(),
            Some(Ty::Uint(UintType::U16)) => "u16".to_string(),
            Some(Ty::Uint(UintType::U32)) => "u32".to_string(),
            Some(Ty::Uint(UintType::U64)) => "u64".to_string(),
            Some(Ty::Uint(UintType::U128)) => "u128".to_string(),
            Some(Ty::Uint(UintType::Usize)) => "usize".to_string(),
            Some(Ty::Float(FloatType::F32)) => "f32".to_string(),
            Some(Ty::Float(FloatType::F64)) => "f64".to_string(),
            Some(Ty::Str) => "str".to_string(),
            Some(Ty::Ref { inner, mutable, .. }) => {
                let m = if mutable { "mut" } else { "" };
                format!("ref{}{}", m, self.mangle_type(inner))
            }
            Some(Ty::Ptr { inner, mutable }) => {
                let m = if mutable { "mut" } else { "const" };
                format!("ptr{}{}", m, self.mangle_type(inner))
            }
            Some(Ty::Slice(inner)) => format!("slice{}", self.mangle_type(inner)),
            Some(Ty::Array { elem, len }) => format!("arr{}x{}", self.mangle_type(elem), len),
            Some(Ty::Tuple(elems)) => {
                let parts: Vec<_> = elems.iter().map(|&e| self.mangle_type(e)).collect();
                format!("tup{}", parts.join("_"))
            }
            Some(Ty::Adt { name, args, .. }) => {
                if args.is_empty() {
                    name
                } else {
                    let parts: Vec<_> = args.iter().map(|&a| self.mangle_type(a)).collect();
                    format!("{}{}", name, parts.join("_"))
                }
            }
            Some(Ty::Param { name, .. }) => name,
            _ => "unknown".to_string(),
        }
    }

    /// Calculate the layout of a type
    pub fn layout(&self, ty_id: TyId) -> TypeLayout {
        // Check cache
        if let Some(layout) = self.layouts.read().get(&ty_id) {
            return layout.clone();
        }

        let layout = self.compute_layout(ty_id);
        self.layouts.write().insert(ty_id, layout.clone());
        layout
    }

    fn compute_layout(&self, ty_id: TyId) -> TypeLayout {
        match self.get(ty_id) {
            Some(Ty::Unit) => TypeLayout {
                size: 0,
                align: 1,
                fields: None,
                variants: None,
            },
            Some(Ty::Bool) => TypeLayout {
                size: 1,
                align: 1,
                fields: None,
                variants: None,
            },
            Some(Ty::Char) => TypeLayout {
                size: 4,
                align: 4,
                fields: None,
                variants: None,
            },
            Some(Ty::Int(IntType::I8)) | Some(Ty::Uint(UintType::U8)) => TypeLayout {
                size: 1,
                align: 1,
                fields: None,
                variants: None,
            },
            Some(Ty::Int(IntType::I16)) | Some(Ty::Uint(UintType::U16)) => TypeLayout {
                size: 2,
                align: 2,
                fields: None,
                variants: None,
            },
            Some(Ty::Int(IntType::I32)) | Some(Ty::Uint(UintType::U32)) => TypeLayout {
                size: 4,
                align: 4,
                fields: None,
                variants: None,
            },
            Some(Ty::Int(IntType::I64))
            | Some(Ty::Int(IntType::Isize))
            | Some(Ty::Uint(UintType::U64))
            | Some(Ty::Uint(UintType::Usize)) => TypeLayout {
                size: 8,
                align: 8,
                fields: None,
                variants: None,
            },
            Some(Ty::Int(IntType::I128)) | Some(Ty::Uint(UintType::U128)) => TypeLayout {
                size: 16,
                align: 16,
                fields: None,
                variants: None,
            },
            Some(Ty::Float(FloatType::F32)) => TypeLayout {
                size: 4,
                align: 4,
                fields: None,
                variants: None,
            },
            Some(Ty::Float(FloatType::F64)) => TypeLayout {
                size: 8,
                align: 8,
                fields: None,
                variants: None,
            },
            Some(Ty::Ref { .. }) | Some(Ty::Ptr { .. }) => TypeLayout {
                size: 8,
                align: 8,
                fields: None,
                variants: None,
            },
            Some(Ty::Slice(_)) | Some(Ty::Str) => {
                // Fat pointer: ptr + len
                TypeLayout {
                    size: 16,
                    align: 8,
                    fields: None,
                    variants: None,
                }
            }
            Some(Ty::Array { elem, len }) => {
                let elem_layout = self.layout(elem);
                TypeLayout {
                    size: elem_layout.size * len,
                    align: elem_layout.align,
                    fields: None,
                    variants: None,
                }
            }
            Some(Ty::Tuple(elems)) => {
                let mut size = 0usize;
                let mut max_align = 1usize;
                let mut fields = IndexMap::new();

                for (i, &elem_id) in elems.iter().enumerate() {
                    let elem_layout = self.layout(elem_id);
                    // Align the offset
                    let offset = (size + elem_layout.align - 1) & !(elem_layout.align - 1);
                    fields.insert(
                        i.to_string(),
                        FieldLayout {
                            offset,
                            ty: elem_id,
                        },
                    );
                    size = offset + elem_layout.size;
                    max_align = max_align.max(elem_layout.align);
                }

                // Align total size
                size = (size + max_align - 1) & !(max_align - 1);

                TypeLayout {
                    size,
                    align: max_align,
                    fields: Some(fields),
                    variants: None,
                }
            }
            Some(Ty::Adt { name, args, .. }) => {
                // Check if struct or enum
                if let Some(struct_def) = self.struct_defs.read().get(&name).cloned() {
                    self.compute_struct_layout(&struct_def, &args)
                } else if let Some(enum_def) = self.enum_defs.read().get(&name).cloned() {
                    self.compute_enum_layout(&enum_def, &args)
                } else {
                    // Unknown ADT, treat as 8-byte value
                    TypeLayout {
                        size: 8,
                        align: 8,
                        fields: None,
                        variants: None,
                    }
                }
            }
            Some(Ty::Fn { .. }) => {
                // Function pointer
                TypeLayout {
                    size: 8,
                    align: 8,
                    fields: None,
                    variants: None,
                }
            }

            // Built-in collection types
            Some(Ty::Vec(_)) => {
                // Vec layout: ptr (8) + len (8) + capacity (8) = 24 bytes
                let mut fields = IndexMap::new();
                fields.insert("ptr".to_string(), FieldLayout { offset: 0, ty: ty_id });
                fields.insert("len".to_string(), FieldLayout { offset: 8, ty: ty_id });
                fields.insert("cap".to_string(), FieldLayout { offset: 16, ty: ty_id });
                TypeLayout {
                    size: 24,
                    align: 8,
                    fields: Some(fields),
                    variants: None,
                }
            }
            Some(Ty::String) => {
                // String layout: same as Vec<u8>
                let mut fields = IndexMap::new();
                fields.insert("ptr".to_string(), FieldLayout { offset: 0, ty: ty_id });
                fields.insert("len".to_string(), FieldLayout { offset: 8, ty: ty_id });
                fields.insert("cap".to_string(), FieldLayout { offset: 16, ty: ty_id });
                TypeLayout {
                    size: 24,
                    align: 8,
                    fields: Some(fields),
                    variants: None,
                }
            }
            Some(Ty::Box(_)) => {
                // Box layout: just a pointer
                TypeLayout {
                    size: 8,
                    align: 8,
                    fields: None,
                    variants: None,
                }
            }
            Some(Ty::Option(inner)) => {
                // Option layout: discriminant (8 for alignment) + inner
                let inner_layout = self.layout(inner);
                let inner_size = inner_layout.size;
                TypeLayout {
                    size: 8 + inner_size,
                    align: 8,
                    fields: None,
                    variants: None,
                }
            }
            Some(Ty::Result { ok, err }) => {
                // Result layout: discriminant (8) + max(ok, err)
                let ok_layout = self.layout(ok);
                let err_layout = self.layout(err);
                let max_size = ok_layout.size.max(err_layout.size);
                TypeLayout {
                    size: 8 + max_size,
                    align: 8,
                    fields: None,
                    variants: None,
                }
            }

            _ => TypeLayout {
                size: 8,
                align: 8,
                fields: None,
                variants: None,
            },
        }
    }

    fn compute_struct_layout(&self, struct_def: &StructDef, args: &[TyId]) -> TypeLayout {
        // Build substitution map
        let mut subst = HashMap::new();
        for (i, param) in struct_def.generics.params.iter().enumerate() {
            if let GenericParam::Type { name, .. } = param {
                if i < args.len() {
                    subst.insert(name.clone(), args[i]);
                }
            }
        }

        let mut size = 0usize;
        let mut max_align = 1usize;
        let mut fields = IndexMap::new();

        for (field_name, field_hir_ty) in &struct_def.fields {
            let field_ty = self.resolve_hir_type_with_subst(field_hir_ty, &subst);
            let field_layout = self.layout(field_ty);

            // Align the offset
            let offset = (size + field_layout.align - 1) & !(field_layout.align - 1);
            fields.insert(
                field_name.clone(),
                FieldLayout {
                    offset,
                    ty: field_ty,
                },
            );
            size = offset + field_layout.size;
            max_align = max_align.max(field_layout.align);
        }

        // Align total size
        if max_align > 0 {
            size = (size + max_align - 1) & !(max_align - 1);
        }

        // Minimum size of 8 for structs (for pointer operations)
        if size == 0 {
            size = 8;
        }

        TypeLayout {
            size,
            align: max_align.max(1),
            fields: Some(fields),
            variants: None,
        }
    }

    fn compute_enum_layout(&self, enum_def: &EnumDef, args: &[TyId]) -> TypeLayout {
        // Build substitution map
        let mut subst = HashMap::new();
        for (i, param) in enum_def.generics.params.iter().enumerate() {
            if let GenericParam::Type { name, .. } = param {
                if i < args.len() {
                    subst.insert(name.clone(), args[i]);
                }
            }
        }

        let mut max_payload = 0usize;
        let mut variants = IndexMap::new();

        for (disc, variant) in enum_def.variants.iter().enumerate() {
            let mut payload_size = 0usize;
            let mut variant_fields = IndexMap::new();
            let mut offset = 0usize;

            for (field_name, field_hir_ty) in &variant.fields {
                let field_ty = self.resolve_hir_type_with_subst(field_hir_ty, &subst);
                let field_layout = self.layout(field_ty);

                // Align within variant
                offset = (offset + field_layout.align - 1) & !(field_layout.align - 1);
                variant_fields.insert(
                    field_name.clone(),
                    FieldLayout {
                        offset,
                        ty: field_ty,
                    },
                );
                offset += field_layout.size;
            }

            payload_size = offset;
            max_payload = max_payload.max(payload_size);

            variants.insert(
                variant.name.clone(),
                VariantLayout {
                    discriminant: disc as i64,
                    fields: if variant_fields.is_empty() {
                        None
                    } else {
                        Some(variant_fields)
                    },
                    payload_size,
                },
            );
        }

        // Enum = 8-byte discriminant + max payload
        TypeLayout {
            size: 8 + max_payload,
            align: 8,
            fields: None,
            variants: Some(variants),
        }
    }

    /// Get struct definition by name
    pub fn get_struct(&self, name: &str) -> Option<StructDef> {
        self.struct_defs.read().get(name).cloned()
    }

    /// Get enum definition by name
    pub fn get_enum(&self, name: &str) -> Option<EnumDef> {
        self.enum_defs.read().get(name).cloned()
    }

    /// Check if a type is a known struct
    pub fn is_struct(&self, name: &str) -> bool {
        self.struct_defs.read().contains_key(name)
    }

    /// Check if a type is a known enum
    pub fn is_enum(&self, name: &str) -> bool {
        self.enum_defs.read().contains_key(name)
    }

    /// Check if a type implements a specific trait
    pub fn type_implements_trait(&self, type_name: &str, trait_name: &str) -> bool {
        let table = self.method_table.read();
        table.get_trait_impls(type_name).contains(&trait_name.to_string())
    }

    /// Get the size of a type in bytes
    pub fn size_of(&self, ty_id: TyId) -> usize {
        self.layout(ty_id).size
    }

    /// Get the alignment of a type in bytes
    pub fn align_of(&self, ty_id: TyId) -> usize {
        self.layout(ty_id).align
    }

    // ========================================================================
    // Expression and Local Type Recording (populated by typeck)
    // ========================================================================

    /// Record the type of an expression by its span
    pub fn record_expr_type(&self, span: Span, ty: TyId) {
        self.expr_types.write().insert(span, ty);
    }

    /// Get the type of an expression by its span
    pub fn get_expr_type(&self, span: Span) -> Option<TyId> {
        self.expr_types.read().get(&span).copied()
    }

    /// Record a local variable's type
    pub fn record_local_type(&self, func: &str, name: &str, ty: TyId) {
        self.local_types.write().insert((func.to_string(), name.to_string()), ty);
    }

    /// Get a local variable's type
    pub fn get_local_type(&self, func: &str, name: &str) -> Option<TyId> {
        self.local_types.read().get(&(func.to_string(), name.to_string())).copied()
    }

    // ========================================================================
    // Method Resolution
    // ========================================================================

    /// Resolve a method call on a receiver type.
    /// Returns the MethodInfo if found, None otherwise.
    ///
    /// Resolution order:
    /// 1. Inherent methods on the exact type
    /// 2. Trait impl methods for traits the type implements
    /// 3. Deref coercion: try the deref target type
    pub fn resolve_method(&self, receiver_ty: TyId, method_name: &str) -> Option<MethodInfo> {
        let type_name = self.type_name_for_method_lookup(receiver_ty);
        self.resolve_method_by_name(&type_name, method_name, 0)
    }

    /// Resolve method with deref depth limit to prevent infinite loops
    /// Public for use by codegen to unify method resolution
    pub fn resolve_method_by_name(&self, type_name: &str, method_name: &str, depth: usize) -> Option<MethodInfo> {
        // Prevent infinite deref chains
        if depth > 10 {
            return None;
        }

        let table = self.method_table.read();

        // 1. Try inherent methods first
        if let Some(info) = table.get_inherent_method(type_name, method_name) {
            return Some(info.clone());
        }

        // 2. Try trait impl methods
        for trait_name in table.get_trait_impls(type_name) {
            if let Some(info) = table.get_trait_impl_method(type_name, trait_name, method_name) {
                return Some(info.clone());
            }
        }

        // 3. Try deref coercion
        if let Some(deref_target) = table.get_deref_target(type_name) {
            let target = deref_target.to_string(); // Clone before dropping lock
            drop(table); // Release lock before recursing
            return self.resolve_method_by_name(&target, method_name, depth + 1);
        }

        None
    }

    /// Get the type name for method resolution
    fn type_name_for_method_lookup(&self, ty_id: TyId) -> String {
        match self.get(ty_id) {
            Some(Ty::Adt { name, .. }) => name,
            Some(Ty::Vec(_)) => "Vec".to_string(),
            Some(Ty::String) => "String".to_string(),
            Some(Ty::Box(_)) => "Box".to_string(),
            Some(Ty::Option(_)) => "Option".to_string(),
            Some(Ty::Result { .. }) => "Result".to_string(),
            Some(Ty::Str) => "str".to_string(),
            Some(Ty::Slice(_)) => "slice".to_string(),
            Some(Ty::Ref { inner, .. }) => {
                // For references, look up methods on the inner type
                self.type_name_for_method_lookup(inner)
            }
            Some(Ty::Ptr { inner, .. }) => {
                self.type_name_for_method_lookup(inner)
            }
            _ => "Unknown".to_string(),
        }
    }

    /// Get the mangled method name for codegen
    pub fn get_method_mangled_name(&self, receiver_ty: TyId, method_name: &str) -> Option<String> {
        self.resolve_method(receiver_ty, method_name)
            .map(|info| info.mangled_name)
    }

    /// Check if a method exists on a type
    pub fn has_method(&self, receiver_ty: TyId, method_name: &str) -> bool {
        self.resolve_method(receiver_ty, method_name).is_some()
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
