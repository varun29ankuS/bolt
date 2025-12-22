//! Unified Type System for Bolt
//!
//! This module provides a single source of truth for types throughout compilation.
//! It handles:
//! - Type definitions (structs, enums, type aliases)
//! - Type resolution (resolving Path types to concrete definitions)
//! - Monomorphization (specializing generic types/functions)
//! - Type layout calculation for codegen

use crate::hir::{
    self, Crate, DefId, Enum, FloatType, GenericParam, Generics, IntType, ItemKind, Path,
    PathSegment, Struct, StructKind, Type as HirType, TypeAlias, TypeKind, UintType,
};
use indexmap::IndexMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// A unique identifier for a resolved type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyId(pub u32);

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
        lifetime: Option<String>,
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
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self {
            types: RwLock::new(Vec::new()),
            intern_map: RwLock::new(HashMap::new()),
            next_id: AtomicU32::new(0),
            next_infer: AtomicU32::new(0),
            struct_defs: RwLock::new(HashMap::new()),
            enum_defs: RwLock::new(HashMap::new()),
            type_aliases: RwLock::new(HashMap::new()),
            layouts: RwLock::new(HashMap::new()),
            mono_instances: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize the registry from a crate's type definitions
    pub fn init_from_crate(&self, krate: &Crate) {
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
                _ => {}
            }
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
                self.intern(Ty::Ref {
                    lifetime: lifetime.clone(),
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
        }
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

            // Check for built-in types
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

    /// Get the size of a type in bytes
    pub fn size_of(&self, ty_id: TyId) -> usize {
        self.layout(ty_id).size
    }

    /// Get the alignment of a type in bytes
    pub fn align_of(&self, ty_id: TyId) -> usize {
        self.layout(ty_id).align
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
