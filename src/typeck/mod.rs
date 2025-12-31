//! Type checking for Bolt
//!
//! Performs type inference and checking with proper type resolution.

use crate::cache::ExprCache;
use crate::error::{Diagnostic, DiagnosticEmitter, Result, Span};
use crate::extern_crates::{global_resolver, ResolvedItem, StubItemKind};
use crate::hir::*;
use crate::ty::{Lifetime, LifetimeId, Ty, TyId, TypeRegistry};
use indexmap::IndexMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Type checking context that holds the type registry and diagnostics
pub struct TypeContext {
    pub registry: Arc<TypeRegistry>,
    diagnostics: RwLock<DiagnosticEmitter>,
    /// Union-Find structure for type inference unification
    unification: RwLock<HashMap<u32, TyId>>,
    /// Expression-level incremental compilation cache
    expr_cache: RwLock<ExprCache>,
}

impl TypeContext {
    pub fn new(registry: Arc<TypeRegistry>) -> Self {
        Self {
            registry,
            diagnostics: RwLock::new(DiagnosticEmitter::new()),
            unification: RwLock::new(HashMap::new()),
            expr_cache: RwLock::new(ExprCache::new()),
        }
    }

    /// Get cached type for an expression by its content hash.
    pub fn get_cached_type(&self, expr_hash: u64) -> Option<TyId> {
        self.expr_cache.write().get_type(expr_hash).map(TyId)
    }

    /// Cache a type checking result for an expression.
    pub fn cache_type(&self, expr_hash: u64, ty: TyId) {
        self.expr_cache.write().insert_type(expr_hash, ty.0);
    }

    /// Get expression cache statistics.
    pub fn expr_cache_stats(&self) -> (u64, u64, f64) {
        let cache = self.expr_cache.read();
        let stats = cache.stats();
        (stats.type_hits, stats.type_misses, stats.type_hit_rate())
    }

    /// Record the type of an expression (delegates to registry)
    pub fn record_expr_type(&self, span: Span, ty: TyId) {
        self.registry.record_expr_type(span, ty);
    }

    /// Get the type of an expression by its span (delegates to registry)
    pub fn get_expr_type(&self, span: Span) -> Option<TyId> {
        self.registry.get_expr_type(span)
    }

    /// Record a local variable's type (delegates to registry)
    pub fn record_local_type(&self, func: &str, name: &str, ty: TyId) {
        self.registry.record_local_type(func, name, ty);
    }

    /// Get a local variable's type (delegates to registry)
    pub fn get_local_type(&self, func: &str, name: &str) -> Option<TyId> {
        self.registry.get_local_type(func, name)
    }

    pub fn emit_error(&self, diag: Diagnostic) {
        self.diagnostics.write().emit(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.read().has_errors()
    }

    pub fn take_diagnostics(&self) -> Vec<Diagnostic> {
        self.diagnostics.write().take_diagnostics()
    }

    /// Find the representative type for an inference variable
    fn find(&self, ty_id: TyId) -> TyId {
        match self.registry.get(ty_id) {
            Some(Ty::Infer(var)) => {
                if let Some(&unified) = self.unification.read().get(&var) {
                    if unified != ty_id {
                        return self.find(unified);
                    }
                }
                ty_id
            }
            _ => ty_id,
        }
    }

    /// Unify two types, updating inference variables
    pub fn unify(&self, a: TyId, b: TyId) -> bool {
        let a = self.find(a);
        let b = self.find(b);

        if a == b {
            return true;
        }

        let ty_a = self.registry.get(a);
        let ty_b = self.registry.get(b);

        match (&ty_a, &ty_b) {
            // Inference variables unify with anything
            (Some(Ty::Infer(var)), _) => {
                self.unification.write().insert(*var, b);
                true
            }
            (_, Some(Ty::Infer(var))) => {
                self.unification.write().insert(*var, a);
                true
            }
            // Error types unify with anything (error recovery)
            (Some(Ty::Error), _) | (_, Some(Ty::Error)) => true,
            // Never type unifies with anything
            (Some(Ty::Never), _) | (_, Some(Ty::Never)) => true,
            // Same primitive types
            (Some(Ty::Unit), Some(Ty::Unit)) => true,
            (Some(Ty::Bool), Some(Ty::Bool)) => true,
            (Some(Ty::Char), Some(Ty::Char)) => true,
            // Numeric coercion: allow different int/uint sizes for self-hosting flexibility
            (Some(Ty::Int(_)), Some(Ty::Int(_))) => true,
            (Some(Ty::Uint(_)), Some(Ty::Uint(_))) => true,
            (Some(Ty::Float(f1)), Some(Ty::Float(f2))) => f1 == f2,
            (Some(Ty::Str), Some(Ty::Str)) => true,
            // References
            (
                Some(Ty::Ref { inner: a_inner, mutable: a_mut, .. }),
                Some(Ty::Ref { inner: b_inner, mutable: b_mut, .. }),
            ) => {
                // Check for array-to-slice coercion: &[T; N] -> &[T]
                if a_mut == b_mut {
                    let inner_a = self.registry.get(*a_inner);
                    let inner_b = self.registry.get(*b_inner);
                    match (&inner_a, &inner_b) {
                        // &[T; N] coerces to &[T]
                        (Some(Ty::Array { elem: a_elem, .. }), Some(Ty::Slice(b_elem))) => {
                            return self.unify(*a_elem, *b_elem);
                        }
                        // &[T] coerces from &[T; N] (reverse direction)
                        (Some(Ty::Slice(a_elem)), Some(Ty::Array { elem: b_elem, .. })) => {
                            return self.unify(*a_elem, *b_elem);
                        }
                        // &T coerces to &dyn Trait if T implements Trait
                        (Some(concrete_ty), Some(Ty::DynTrait { bounds })) => {
                            return self.check_type_implements_traits(*a_inner, &concrete_ty, bounds);
                        }
                        // &dyn Trait accepts &T if T implements Trait (reverse)
                        (Some(Ty::DynTrait { bounds }), Some(concrete_ty)) => {
                            return self.check_type_implements_traits(*b_inner, &concrete_ty, bounds);
                        }
                        // Deref coercion: &String -> &str
                        (Some(Ty::Str), Some(Ty::String)) => {
                            return true;
                        }
                        // Deref coercion: &str <- &String (reverse)
                        (Some(Ty::String), Some(Ty::Str)) => {
                            return true;
                        }
                        // Also handle Adt version of String
                        (Some(Ty::Str), Some(Ty::Adt { name, .. })) if name == "String" => {
                            return true;
                        }
                        (Some(Ty::Adt { name, .. }), Some(Ty::Str)) if name == "String" => {
                            return true;
                        }
                        // Deref coercion: &Box<T> -> &T (Box is Ty::Box, not Ty::Adt)
                        (Some(_), Some(Ty::Box(box_inner))) => {
                            // &Box<T> can be dereferenced to &T
                            if self.unify(*a_inner, *box_inner) {
                                return true;
                            }
                        }
                        // Deref coercion: &Vec<T> -> &[T] (Vec is Ty::Vec, not Ty::Adt)
                        (Some(Ty::Slice(slice_elem)), Some(Ty::Vec(vec_elem))) => {
                            if self.unify(*slice_elem, *vec_elem) {
                                return true;
                            }
                        }
                        // Deref coercion: &[T] <- &Vec<T> (reverse)
                        (Some(Ty::Vec(vec_elem)), Some(Ty::Slice(slice_elem))) => {
                            if self.unify(*vec_elem, *slice_elem) {
                                return true;
                            }
                        }
                        // Also handle Adt version for backwards compat
                        (Some(Ty::Slice(slice_elem)), Some(Ty::Adt { name, args, .. })) if name == "Vec" && !args.is_empty() => {
                            if self.unify(*slice_elem, args[0]) {
                                return true;
                            }
                        }
                        (Some(Ty::Adt { name, args, .. }), Some(Ty::Slice(slice_elem))) if name == "Vec" && !args.is_empty() => {
                            if self.unify(args[0], *slice_elem) {
                                return true;
                            }
                        }
                        // Deref coercion: &T <- &Box<T> (reverse)
                        (Some(Ty::Box(box_inner)), Some(_)) => {
                            if self.unify(*box_inner, *b_inner) {
                                return true;
                            }
                        }
                        // Also handle Adt version for backwards compatibility
                        (Some(_), Some(Ty::Adt { name, args, .. })) if name == "Box" && !args.is_empty() => {
                            if self.unify(*a_inner, args[0]) {
                                return true;
                            }
                        }
                        (Some(Ty::Adt { name, args, .. }), Some(_)) if name == "Box" && !args.is_empty() => {
                            if self.unify(args[0], *b_inner) {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
                a_mut == b_mut && self.unify(*a_inner, *b_inner)
            }
            // Pointers
            (
                Some(Ty::Ptr { inner: a_inner, mutable: a_mut }),
                Some(Ty::Ptr { inner: b_inner, mutable: b_mut }),
            ) => a_mut == b_mut && self.unify(*a_inner, *b_inner),
            // Slices
            (Some(Ty::Slice(a_inner)), Some(Ty::Slice(b_inner))) => {
                self.unify(*a_inner, *b_inner)
            }
            // Vec
            (Some(Ty::Vec(a_inner)), Some(Ty::Vec(b_inner))) => {
                self.unify(*a_inner, *b_inner)
            }
            // Box
            (Some(Ty::Box(a_inner)), Some(Ty::Box(b_inner))) => {
                self.unify(*a_inner, *b_inner)
            }
            // Arrays
            (
                Some(Ty::Array { elem: a_elem, len: a_len }),
                Some(Ty::Array { elem: b_elem, len: b_len }),
            ) => a_len == b_len && self.unify(*a_elem, *b_elem),
            // Tuples
            (Some(Ty::Tuple(a_elems)), Some(Ty::Tuple(b_elems))) => {
                if a_elems.len() != b_elems.len() {
                    return false;
                }
                a_elems
                    .iter()
                    .zip(b_elems.iter())
                    .all(|(&a, &b)| self.unify(a, b))
            }
            // Functions
            (
                Some(Ty::Fn { inputs: a_in, output: a_out }),
                Some(Ty::Fn { inputs: b_in, output: b_out }),
            ) => {
                if a_in.len() != b_in.len() {
                    return false;
                }
                a_in.iter().zip(b_in.iter()).all(|(&a, &b)| self.unify(a, b))
                    && self.unify(*a_out, *b_out)
            }
            // ADTs (structs/enums)
            (
                Some(Ty::Adt { def_id: a_def, args: a_args, .. }),
                Some(Ty::Adt { def_id: b_def, args: b_args, .. }),
            ) => {
                if a_def != b_def || a_args.len() != b_args.len() {
                    return false;
                }
                a_args
                    .iter()
                    .zip(b_args.iter())
                    .all(|(&a, &b)| self.unify(a, b))
            }
            // Type parameters - unify if same name and index
            (
                Some(Ty::Param { name: a_name, index: a_idx }),
                Some(Ty::Param { name: b_name, index: b_idx }),
            ) => a_name == b_name && a_idx == b_idx,
            // Type parameter unifies with any type (for now - proper constraint checking later)
            (Some(Ty::Param { .. }), _) | (_, Some(Ty::Param { .. })) => true,
            // impl Trait unifies with concrete types that implement the trait(s)
            (Some(Ty::ImplTrait { bounds, .. }), Some(concrete_ty)) => {
                // Check that concrete type implements all required traits
                self.check_type_implements_traits(b, &concrete_ty, bounds)
            }
            (Some(concrete_ty), Some(Ty::ImplTrait { bounds, .. })) => {
                // Check that concrete type implements all required traits
                self.check_type_implements_traits(a, &concrete_ty, bounds)
            }
            // dyn Trait types - same trait bounds must match
            (Some(Ty::DynTrait { bounds: a_bounds }), Some(Ty::DynTrait { bounds: b_bounds })) => {
                a_bounds == b_bounds
            }
            // Box<T> coercion: accept Box<T> where T expected (via Deref) - Ty::Box variant
            (Some(_), Some(Ty::Box(box_inner))) => {
                self.unify(a, *box_inner)
            }
            // Box<T> coercion: accept Box<T> where T expected (via Deref) - Ty::Adt variant
            (Some(_), Some(Ty::Adt { name, args, .. })) if name == "Box" && !args.is_empty() => {
                self.unify(a, args[0])
            }
            // Box<T> coercion: accept Box<T> where &T expected (via Deref + auto-ref) - Ty::Box
            (Some(Ty::Ref { inner, mutable: false, .. }), Some(Ty::Box(box_inner))) => {
                self.unify(*inner, *box_inner)
            }
            // Box<T> coercion: accept Box<T> where &T expected - Ty::Adt variant
            (Some(Ty::Ref { inner, mutable: false, .. }), Some(Ty::Adt { name, args, .. })) if name == "Box" && !args.is_empty() => {
                self.unify(*inner, args[0])
            }
            // Auto-ref coercion: expected &T, got T -> automatically take reference
            // This must come last to avoid matching more specific cases
            (Some(Ty::Ref { inner, mutable: false, .. }), Some(actual)) => {
                // Only auto-ref for non-reference types
                if !matches!(actual, Ty::Ref { .. }) {
                    self.unify(*inner, b)
                } else {
                    false
                }
            }
            // Numeric coercion: i64 <-> usize (common in Bolt codegen)
            (Some(Ty::Int(_)), Some(Ty::Uint(_))) | (Some(Ty::Uint(_)), Some(Ty::Int(_))) => true,
            // Pointer deref coercion: expected T, got *const T or *mut T -> auto-deref
            (Some(_), Some(Ty::Ptr { inner, .. })) => {
                self.unify(a, *inner)
            }
            // Pointer deref coercion reverse: expected *const T or *mut T, got T
            (Some(Ty::Ptr { inner, .. }), Some(_)) => {
                self.unify(*inner, b)
            }
            // Unit coercion: () unifies with anything for statement-like expressions
            // This allows if-else branches to have different types when result isn't used
            (Some(Ty::Unit), _) | (_, Some(Ty::Unit)) => true,
            // Tuple coercion: allow tuples to unify with unit for statement-like expressions
            (Some(Ty::Tuple(_)), Some(Ty::Unit)) | (Some(Ty::Unit), Some(Ty::Tuple(_))) => true,
            _ => false,
        }
    }

    /// Check if a concrete type implements all the given trait bounds
    fn check_type_implements_traits(&self, ty_id: TyId, ty: &Ty, bounds: &[String]) -> bool {
        // Get the type name for trait implementation lookup
        let type_name = match ty {
            Ty::Adt { name, .. } => name.clone(),
            Ty::Int(_) => "i64".to_string(),
            Ty::Uint(_) => "u64".to_string(),
            Ty::Float(_) => "f64".to_string(),
            Ty::Bool => "bool".to_string(),
            Ty::Char => "char".to_string(),
            Ty::String => "String".to_string(),
            Ty::Vec(_) => "Vec".to_string(),
            Ty::Box(_) => "Box".to_string(),
            Ty::Option(_) => "Option".to_string(),
            Ty::Result { .. } => "Result".to_string(),
            _ => return true, // For unknown types, be permissive for now
        };

        // Check each required trait bound
        for trait_name in bounds {
            if !self.registry.type_implements_trait(&type_name, trait_name) {
                return false;
            }
        }
        true
    }

    /// Get the resolved type, following unification
    pub fn resolve(&self, ty_id: TyId) -> TyId {
        self.find(ty_id)
    }

    /// Pretty print a type for error messages
    pub fn display_type(&self, ty_id: TyId) -> String {
        let ty_id = self.find(ty_id);
        match self.registry.get(ty_id) {
            Some(Ty::Unit) => "()".to_string(),
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
            Some(Ty::Ref { inner, mutable, lifetime }) => {
                let m = if mutable { "mut " } else { "" };
                let lt = self.display_lifetime(lifetime);
                if lt.is_empty() {
                    format!("&{}{}", m, self.display_type(inner))
                } else {
                    format!("&{} {}{}", lt, m, self.display_type(inner))
                }
            }
            Some(Ty::Ptr { inner, mutable }) => {
                let m = if mutable { "mut" } else { "const" };
                format!("*{} {}", m, self.display_type(inner))
            }
            Some(Ty::Slice(inner)) => format!("[{}]", self.display_type(inner)),
            Some(Ty::Array { elem, len }) => format!("[{}; {}]", self.display_type(elem), len),
            Some(Ty::Tuple(elems)) => {
                let parts: Vec<_> = elems.iter().map(|&e| self.display_type(e)).collect();
                format!("({})", parts.join(", "))
            }
            Some(Ty::Fn { inputs, output }) => {
                let params: Vec<_> = inputs.iter().map(|&i| self.display_type(i)).collect();
                format!("fn({}) -> {}", params.join(", "), self.display_type(output))
            }
            Some(Ty::Adt { name, args, .. }) => {
                if args.is_empty() {
                    name
                } else {
                    let params: Vec<_> = args.iter().map(|&a| self.display_type(a)).collect();
                    format!("{}<{}>", name, params.join(", "))
                }
            }
            Some(Ty::Never) => "!".to_string(),
            Some(Ty::Infer(var)) => format!("?{}", var),
            Some(Ty::Param { name, .. }) => name,
            Some(Ty::Error) => "<error>".to_string(),
            // Built-in collection types
            Some(Ty::Vec(elem)) => format!("Vec<{}>", self.display_type(elem)),
            Some(Ty::String) => "String".to_string(),
            Some(Ty::Box(inner)) => format!("Box<{}>", self.display_type(inner)),
            Some(Ty::Option(inner)) => format!("Option<{}>", self.display_type(inner)),
            Some(Ty::Result { ok, err }) => {
                format!("Result<{}, {}>", self.display_type(ok), self.display_type(err))
            }
            Some(Ty::ImplTrait { bounds, concrete }) => {
                if let Some(concrete_ty) = concrete {
                    self.display_type(concrete_ty)
                } else if bounds.is_empty() {
                    "impl ?".to_string()
                } else {
                    format!("impl {}", bounds.join(" + "))
                }
            }
            Some(Ty::DynTrait { bounds }) => {
                if bounds.is_empty() {
                    "dyn ?".to_string()
                } else {
                    format!("dyn {}", bounds.join(" + "))
                }
            }
            None => "<unknown>".to_string(),
        }
    }

    /// Display a lifetime for error messages
    pub fn display_lifetime(&self, lifetime_id: LifetimeId) -> String {
        match self.registry.get_lifetime(lifetime_id) {
            Some(Lifetime::Static) => "'static".to_string(),
            Some(Lifetime::Named(name)) => format!("'{}", name),
            Some(Lifetime::Infer(_)) => String::new(), // Hide inference vars
            Some(Lifetime::Anonymous) => String::new(), // Hide anonymous
            Some(Lifetime::Bound { .. }) => String::new(), // Hide bound
            None => String::new(),
        }
    }
}

impl Default for TypeContext {
    fn default() -> Self {
        Self::new(Arc::new(TypeRegistry::new()))
    }
}

/// Type checker for a single function
pub struct TypeChecker<'a> {
    ctx: &'a TypeContext,
    krate: &'a Crate,
    /// Local variable types
    locals: IndexMap<String, TyId>,
    /// Generic type parameters in scope
    type_params: HashMap<String, TyId>,
    /// Lifetime parameters in scope (e.g., 'a, 'b)
    lifetime_params: HashMap<String, LifetimeId>,
    /// Expected return type of current function
    return_type: Option<TyId>,
    /// Current function name (for recording local types)
    current_function: String,
}

impl<'a> TypeChecker<'a> {
    pub fn new(ctx: &'a TypeContext, krate: &'a Crate) -> Self {
        Self {
            ctx,
            krate,
            locals: IndexMap::new(),
            type_params: HashMap::new(),
            lifetime_params: HashMap::new(),
            return_type: None,
            current_function: String::new(),
        }
    }

    pub fn check_crate(&mut self) -> Result<()> {
        // Collect DefIds that are impl methods - they'll be checked via check_impl
        let impl_method_ids: std::collections::HashSet<DefId> = self.krate.items
            .values()
            .filter_map(|item| {
                if let ItemKind::Impl(impl_block) = &item.kind {
                    Some(impl_block.items.iter().copied())
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        for (def_id, item) in &self.krate.items {
            // Skip impl methods - they're checked via check_impl with proper type params
            if impl_method_ids.contains(def_id) {
                continue;
            }
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
            ItemKind::Macro(_) => Ok(()), // Macros are expanded before type checking
        }
    }

    fn check_function(&mut self, f: &Function, name: &str) -> Result<()> {
        self.locals.clear();
        // DON'T clear type_params or lifetime_params - preserve impl-level params
        self.current_function = name.to_string();

        // Register function-level lifetime parameters
        for param in f.sig.generics.params.iter() {
            if let GenericParam::Lifetime { name: lt_name } = param {
                let lifetime_id = self.ctx.registry.named_lifetime(lt_name);
                self.lifetime_params.insert(lt_name.clone(), lifetime_id);
            }
        }

        // Register function-level type parameters (may override impl params with same name)
        let base_index = self.type_params.len();
        for (i, param) in f.sig.generics.params.iter().enumerate() {
            if let GenericParam::Type { name: param_name, .. } = param {
                let ty_id = self.ctx.registry.intern(Ty::Param {
                    name: param_name.clone(),
                    index: base_index + i,
                });
                self.type_params.insert(param_name.clone(), ty_id);
            }
        }

        // Register parameters and record their types
        for (param_name, param_ty) in &f.sig.inputs {
            let ty_id = self.resolve_type(param_ty);
            self.locals.insert(param_name.clone(), ty_id);
            self.ctx.record_local_type(name, param_name, ty_id);
        }

        let expected_return = self.resolve_type(&f.sig.output);
        self.return_type = Some(expected_return);

        if let Some(ref body) = f.body {
            let actual_return = self.check_block(body)?;
            if !self.ctx.unify(expected_return, actual_return) {
                self.ctx.emit_error(
                    Diagnostic::error(format!(
                        "Function `{}` returns {} but body has type {}",
                        name,
                        self.ctx.display_type(expected_return),
                        self.ctx.display_type(actual_return)
                    ))
                    .with_span(body.span),
                );
            }
        }

        self.return_type = None;
        Ok(())
    }

    fn resolve_type(&self, ty: &Type) -> TyId {
        self.resolve_type_with_lifetimes(ty, &self.type_params, &self.lifetime_params)
    }

    /// Resolve a HIR type with both type and lifetime substitutions
    fn resolve_type_with_lifetimes(
        &self,
        ty: &Type,
        type_subst: &HashMap<String, TyId>,
        lifetime_subst: &HashMap<String, LifetimeId>,
    ) -> TyId {
        match &ty.kind {
            TypeKind::Ref { lifetime, mutable, inner } => {
                let inner_id = self.resolve_type_with_lifetimes(inner, type_subst, lifetime_subst);
                // Resolve lifetime from scope or create fresh
                let lifetime_id = match lifetime {
                    Some(name) if name == "static" => self.ctx.registry.static_lifetime(),
                    Some(name) => {
                        // Check if lifetime is in scope
                        lifetime_subst.get(name)
                            .copied()
                            .unwrap_or_else(|| self.ctx.registry.named_lifetime(name))
                    }
                    None => self.ctx.registry.anonymous_lifetime(),
                };
                self.ctx.registry.intern(Ty::Ref {
                    lifetime: lifetime_id,
                    mutable: *mutable,
                    inner: inner_id,
                })
            }
            // For other types, delegate to registry's resolve
            _ => self.ctx.registry.resolve_hir_type_with_subst(ty, type_subst),
        }
    }

    fn check_block(&mut self, block: &Block) -> Result<TyId> {
        for stmt in &block.stmts {
            self.check_stmt(stmt)?;
        }

        if let Some(ref expr) = block.expr {
            self.check_expr(expr)
        } else {
            Ok(self.ctx.registry.intern(Ty::Unit))
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match &stmt.kind {
            StmtKind::Let { pattern, ty, init } => {
                let init_ty = if let Some(init) = init {
                    self.check_expr(init)?
                } else {
                    self.ctx.registry.fresh_infer()
                };

                let declared_ty = ty.as_ref().map(|t| self.resolve_type(t));

                let final_ty = if let Some(declared) = declared_ty {
                    if !self.ctx.unify(declared, init_ty) {
                        self.ctx.emit_error(
                            Diagnostic::error(format!(
                                "Type mismatch: expected {}, found {}",
                                self.ctx.display_type(declared),
                                self.ctx.display_type(init_ty)
                            ))
                            .with_span(stmt.span),
                        );
                    }
                    declared
                } else {
                    init_ty
                };

                self.bind_pattern(pattern, final_ty);
                Ok(())
            }
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
                self.check_expr(expr)?;
                Ok(())
            }
            StmtKind::Item(_) => Ok(()),
        }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, ty: TyId) {
        match &pattern.kind {
            PatternKind::Ident { name, .. } => {
                self.locals.insert(name.clone(), ty);
                self.ctx.record_local_type(&self.current_function, name, ty);
            }
            PatternKind::Tuple(pats) => {
                if let Some(Ty::Tuple(elem_tys)) = self.ctx.registry.get(ty) {
                    for (pat, &elem_ty) in pats.iter().zip(elem_tys.iter()) {
                        self.bind_pattern(pat, elem_ty);
                    }
                }
            }
            PatternKind::Struct { path, fields, .. } => {
                // Resolve struct type and bind field patterns
                let struct_name = path.segments.last().map(|s| s.ident.as_str()).unwrap_or("");
                if let Some(struct_def) = self.ctx.registry.get_struct(struct_name) {
                    // Get type arguments from the ADT type
                    let type_args = if let Some(Ty::Adt { args, .. }) = self.ctx.registry.get(ty) {
                        args
                    } else {
                        vec![]
                    };

                    // Build substitution
                    let mut subst = HashMap::new();
                    for (i, param) in struct_def.generics.params.iter().enumerate() {
                        if let GenericParam::Type { name, .. } = param {
                            if i < type_args.len() {
                                subst.insert(name.clone(), type_args[i]);
                            }
                        }
                    }

                    for field_pat in fields {
                        if let Some((_, field_ty)) = struct_def.fields.iter().find(|(n, _)| n == &field_pat.name) {
                            let resolved_ty = self.ctx.registry.resolve_hir_type_with_subst(field_ty, &subst);
                            self.bind_pattern(&field_pat.pattern, resolved_ty);
                        }
                    }
                }
            }
            PatternKind::TupleStruct { path, elems } => {
                // Handle both tuple structs and enum variants
                // For EnumName::Variant(x), path has 2 segments: [EnumName, Variant]
                // For TupleStruct(x), path has 1 segment: [TupleStruct]

                let type_args = if let Some(Ty::Adt { args, .. }) = self.ctx.registry.get(ty) {
                    args
                } else {
                    vec![]
                };

                if path.segments.len() >= 2 {
                    // Likely an enum variant: EnumName::Variant
                    let enum_name = path.segments[path.segments.len() - 2].ident.as_str();
                    let variant_name = path.segments.last().map(|s| s.ident.as_str()).unwrap_or("");

                    if let Some(enum_def) = self.ctx.registry.get_enum(enum_name) {
                        let mut subst = HashMap::new();
                        for (i, param) in enum_def.generics.params.iter().enumerate() {
                            if let GenericParam::Type { name, .. } = param {
                                if i < type_args.len() {
                                    subst.insert(name.clone(), type_args[i]);
                                }
                            }
                        }

                        if let Some(variant) = enum_def.variants.iter().find(|v| v.name == variant_name) {
                            for (i, elem_pat) in elems.iter().enumerate() {
                                if let Some((_, field_ty)) = variant.fields.get(i) {
                                    let resolved_ty = self.ctx.registry.resolve_hir_type_with_subst(field_ty, &subst);
                                    self.bind_pattern(elem_pat, resolved_ty);
                                }
                            }
                        }
                    }
                } else {
                    // Single segment - try as tuple struct
                    let type_name = path.segments.last().map(|s| s.ident.as_str()).unwrap_or("");
                    if let Some(struct_def) = self.ctx.registry.get_struct(type_name) {
                        let mut subst = HashMap::new();
                        for (i, param) in struct_def.generics.params.iter().enumerate() {
                            if let GenericParam::Type { name, .. } = param {
                                if i < type_args.len() {
                                    subst.insert(name.clone(), type_args[i]);
                                }
                            }
                        }

                        for (i, elem_pat) in elems.iter().enumerate() {
                            if let Some((_, field_ty)) = struct_def.fields.get(i) {
                                let resolved_ty = self.ctx.registry.resolve_hir_type_with_subst(field_ty, &subst);
                                self.bind_pattern(elem_pat, resolved_ty);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn check_expr(&mut self, expr: &Expr) -> Result<TyId> {
        // Check expression cache for incremental compilation
        // Only use cache for expressions that don't depend on local context
        // (literals, some binary ops, etc.) - expressions with paths may
        // resolve differently in different scopes
        let can_cache = expr.content_hash != 0 && self.is_cacheable_expr(&expr.kind);

        if can_cache {
            if let Some(cached_ty) = self.ctx.get_cached_type(expr.content_hash) {
                // Cache hit - still record for codegen but skip type checking
                self.ctx.record_expr_type(expr.span, cached_ty);
                return Ok(cached_ty);
            }
        }

        // Cache miss or non-cacheable - perform type checking
        let ty = self.check_expr_inner(expr)?;

        // Cache the result for future use
        if can_cache {
            self.ctx.cache_type(expr.content_hash, ty);
        }

        // Record the expression type for codegen to use
        self.ctx.record_expr_type(expr.span, ty);
        Ok(ty)
    }

    /// Determine if an expression can be safely cached.
    /// Expressions that depend on local variable resolution cannot be cached
    /// because the same expression structure may type differently in different scopes.
    fn is_cacheable_expr(&self, kind: &ExprKind) -> bool {
        match kind {
            // Literals are always cacheable
            ExprKind::Lit(_) => true,
            // Paths depend on scope - not cacheable
            ExprKind::Path(_) => false,
            // Binary/unary ops on cacheable subexpressions
            ExprKind::Binary { lhs, rhs, .. } => {
                self.is_cacheable_expr(&lhs.kind) && self.is_cacheable_expr(&rhs.kind)
            }
            ExprKind::Unary { expr, .. } => self.is_cacheable_expr(&expr.kind),
            // Tuples/arrays of cacheable expressions
            ExprKind::Tuple(exprs) | ExprKind::Array(exprs) => {
                exprs.iter().all(|e| self.is_cacheable_expr(&e.kind))
            }
            // Most other expressions involve paths or local state
            _ => false,
        }
    }

    fn check_expr_inner(&mut self, expr: &Expr) -> Result<TyId> {
        match &expr.kind {
            ExprKind::Lit(lit) => Ok(self.literal_type(lit)),
            ExprKind::Path(path) => self.resolve_path(path, expr.span),

            ExprKind::Unary { op, expr: inner } => {
                let inner_ty = self.check_expr(inner)?;
                match op {
                    UnaryOp::Not => {
                        let bool_ty = self.ctx.registry.intern(Ty::Bool);
                        if self.ctx.unify(inner_ty, bool_ty) {
                            Ok(bool_ty)
                        } else {
                            // Bitwise not on integers
                            Ok(inner_ty)
                        }
                    }
                    UnaryOp::Neg => Ok(inner_ty),
                }
            }

            ExprKind::Binary { op, lhs, rhs } => {
                let lhs_ty = self.check_expr(lhs)?;
                let rhs_ty = self.check_expr(rhs)?;
                self.check_binop(*op, lhs_ty, rhs_ty, expr.span)
            }

            ExprKind::Assign { lhs, rhs } => {
                let lhs_ty = self.check_expr(lhs)?;
                let rhs_ty = self.check_expr(rhs)?;
                if !self.ctx.unify(lhs_ty, rhs_ty) {
                    self.ctx.emit_error(
                        Diagnostic::error(format!(
                            "Cannot assign {} to {}",
                            self.ctx.display_type(rhs_ty),
                            self.ctx.display_type(lhs_ty)
                        ))
                        .with_span(expr.span),
                    );
                }
                Ok(self.ctx.registry.intern(Ty::Unit))
            }

            ExprKind::AssignOp { op, lhs, rhs } => {
                let lhs_ty = self.check_expr(lhs)?;
                let rhs_ty = self.check_expr(rhs)?;
                // Allow pointer arithmetic and other low-level ops involving pointers
                let lhs_resolved = self.ctx.registry.get(self.ctx.resolve(lhs_ty));
                let rhs_resolved = self.ctx.registry.get(self.ctx.resolve(rhs_ty));
                let involves_ptr = matches!(&lhs_resolved, Some(Ty::Ptr { .. }))
                    || matches!(&rhs_resolved, Some(Ty::Ptr { .. }));
                if !involves_ptr && !self.ctx.unify(lhs_ty, rhs_ty) {
                    self.ctx.emit_error(
                        Diagnostic::error(format!(
                            "Cannot apply {:?}= with {} and {}",
                            op,
                            self.ctx.display_type(lhs_ty),
                            self.ctx.display_type(rhs_ty)
                        ))
                        .with_span(expr.span),
                    );
                }
                Ok(self.ctx.registry.intern(Ty::Unit))
            }

            ExprKind::Index { expr: base, index } => {
                let base_ty = self.check_expr(base)?;
                let _index_ty = self.check_expr(index)?;

                // Determine element type based on base type
                match self.ctx.registry.get(self.ctx.resolve(base_ty)) {
                    Some(Ty::Array { elem, .. }) => Ok(elem),
                    Some(Ty::Slice(elem)) => Ok(elem),
                    Some(Ty::Ref { inner, .. }) => {
                        // Deref and try again
                        match self.ctx.registry.get(inner) {
                            Some(Ty::Array { elem, .. }) => Ok(elem),
                            Some(Ty::Slice(elem)) => Ok(elem),
                            _ => Ok(self.ctx.registry.fresh_infer()),
                        }
                    }
                    _ => Ok(self.ctx.registry.fresh_infer()),
                }
            }

            ExprKind::Field { expr: base, field } => {
                let base_ty = self.check_expr(base)?;
                self.check_field_access(base_ty, field, expr.span)
            }

            ExprKind::Call { func, args } => {
                let func_ty = self.check_expr(func)?;
                self.check_call(func_ty, args, expr.span)
            }

            ExprKind::MethodCall { receiver, method, args } => {
                let receiver_ty = self.check_expr(receiver)?;
                self.check_method_call(receiver_ty, method, args, expr.span)
            }

            ExprKind::Tuple(elems) => {
                let elem_tys: Vec<_> = elems
                    .iter()
                    .map(|e| self.check_expr(e))
                    .collect::<Result<_>>()?;
                Ok(self.ctx.registry.intern(Ty::Tuple(elem_tys)))
            }

            ExprKind::Array(elems) => {
                if elems.is_empty() {
                    let elem_ty = self.ctx.registry.fresh_infer();
                    Ok(self.ctx.registry.intern(Ty::Array { elem: elem_ty, len: 0 }))
                } else {
                    let first_ty = self.check_expr(&elems[0])?;
                    for elem in &elems[1..] {
                        let elem_ty = self.check_expr(elem)?;
                        if !self.ctx.unify(first_ty, elem_ty) {
                            self.ctx.emit_error(
                                Diagnostic::error("Array elements must have the same type")
                                    .with_span(elem.span),
                            );
                        }
                    }
                    Ok(self.ctx.registry.intern(Ty::Array {
                        elem: first_ty,
                        len: elems.len(),
                    }))
                }
            }

            ExprKind::Repeat { elem, count } => {
                let elem_ty = self.check_expr(elem)?;
                let _count_ty = self.check_expr(count)?;
                // For now, assume count evaluates to a constant
                Ok(self.ctx.registry.intern(Ty::Array { elem: elem_ty, len: 0 }))
            }

            ExprKind::Struct { path, fields, rest } => {
                self.check_struct_expr(path, fields, rest.as_deref(), expr.span)
            }

            ExprKind::If { cond, then_branch, else_branch } => {
                let cond_ty = self.check_expr(cond)?;
                let bool_ty = self.ctx.registry.intern(Ty::Bool);
                if !self.ctx.unify(cond_ty, bool_ty) {
                    self.ctx.emit_error(
                        Diagnostic::error("If condition must be bool")
                            .with_span(cond.span),
                    );
                }

                let then_ty = self.check_block(then_branch)?;

                if let Some(else_expr) = else_branch {
                    let else_ty = self.check_expr(else_expr)?;
                    if !self.ctx.unify(then_ty, else_ty) {
                        self.ctx.emit_error(
                            Diagnostic::error(format!(
                                "If branches have incompatible types: {} vs {}",
                                self.ctx.display_type(then_ty),
                                self.ctx.display_type(else_ty)
                            ))
                            .with_span(expr.span),
                        );
                    }
                    Ok(then_ty)
                } else {
                    Ok(self.ctx.registry.intern(Ty::Unit))
                }
            }

            ExprKind::IfLet { pattern, expr: match_expr, then_branch, else_branch } => {
                let match_ty = self.check_expr(match_expr)?;
                self.bind_pattern(pattern, match_ty);
                let then_ty = self.check_block(then_branch)?;

                if let Some(else_expr) = else_branch {
                    let else_ty = self.check_expr(else_expr)?;
                    if !self.ctx.unify(then_ty, else_ty) {
                        self.ctx.emit_error(
                            Diagnostic::error("If-let branches have incompatible types")
                                .with_span(expr.span),
                        );
                    }
                }
                Ok(then_ty)
            }

            ExprKind::Match { expr: match_expr, arms } => {
                let scrutinee_ty = self.check_expr(match_expr)?;
                let mut result_ty = None;

                for arm in arms {
                    self.bind_pattern(&arm.pattern, scrutinee_ty);
                    if let Some(ref guard) = arm.guard {
                        self.check_expr(guard)?;
                    }
                    let arm_ty = self.check_expr(&arm.body)?;

                    if let Some(prev_ty) = result_ty {
                        if !self.ctx.unify(prev_ty, arm_ty) {
                            self.ctx.emit_error(
                                Diagnostic::error("Match arms have incompatible types")
                                    .with_span(arm.body.span),
                            );
                        }
                    } else {
                        result_ty = Some(arm_ty);
                    }
                }

                Ok(result_ty.unwrap_or_else(|| self.ctx.registry.intern(Ty::Never)))
            }

            ExprKind::Loop { body, .. } => {
                self.check_block(body)?;
                // Loops return Never unless broken out of
                Ok(self.ctx.registry.intern(Ty::Never))
            }

            ExprKind::While { cond, body, .. } => {
                let cond_ty = self.check_expr(cond)?;
                let bool_ty = self.ctx.registry.intern(Ty::Bool);
                if !self.ctx.unify(cond_ty, bool_ty) {
                    self.ctx.emit_error(
                        Diagnostic::error("While condition must be bool")
                            .with_span(cond.span),
                    );
                }
                self.check_block(body)?;
                Ok(self.ctx.registry.intern(Ty::Unit))
            }

            ExprKind::WhileLet { pattern, expr: match_expr, body, .. } => {
                let match_ty = self.check_expr(match_expr)?;
                self.bind_pattern(pattern, match_ty);
                self.check_block(body)?;
                Ok(self.ctx.registry.intern(Ty::Unit))
            }

            ExprKind::For { pattern, iter, body, .. } => {
                let iter_ty = self.check_expr(iter)?;
                // Extract element type from iterator (Vec<T>, [T; N], &[T], Range, etc.)
                let resolved = self.ctx.resolve(iter_ty);
                let elem_ty = match self.ctx.registry.get(resolved) {
                    // Vec<T> -> T (dedicated Vec type)
                    Some(Ty::Vec(elem)) => elem,
                    // Vec<T> -> T (as ADT)
                    Some(Ty::Adt { name, args, .. }) if name == "Vec" && !args.is_empty() => args[0],
                    // [T; N] -> T
                    Some(Ty::Array { elem, .. }) => elem,
                    // &[T] or &Vec<T> -> T
                    Some(Ty::Ref { inner, .. }) => {
                        let inner_resolved = self.ctx.resolve(inner);
                        match self.ctx.registry.get(inner_resolved) {
                            Some(Ty::Vec(elem)) => elem,
                            Some(Ty::Adt { name, args, .. }) if name == "Vec" && !args.is_empty() => args[0],
                            Some(Ty::Array { elem, .. }) => elem,
                            Some(Ty::Slice(elem)) => elem,
                            _ => inner  // Fall back to inner type
                        }
                    }
                    // Range types iterate over their bounds
                    Some(Ty::Adt { name, args, .. }) if name.contains("Range") && !args.is_empty() => args[0],
                    // Slice [T] -> T
                    Some(Ty::Slice(elem)) => elem,
                    // Default: use the iterator type itself
                    _ => iter_ty,
                };
                self.bind_pattern(pattern, elem_ty);
                self.check_block(body)?;
                Ok(self.ctx.registry.intern(Ty::Unit))
            }

            ExprKind::Block(block) => self.check_block(block),

            ExprKind::Return(inner) => {
                if let Some(inner) = inner {
                    let return_ty = self.check_expr(inner)?;
                    if let Some(expected) = self.return_type {
                        if !self.ctx.unify(expected, return_ty) {
                            self.ctx.emit_error(
                                Diagnostic::error(format!(
                                    "Return type mismatch: expected {}, found {}",
                                    self.ctx.display_type(expected),
                                    self.ctx.display_type(return_ty)
                                ))
                                .with_span(expr.span),
                            );
                        }
                    }
                }
                Ok(self.ctx.registry.intern(Ty::Never))
            }

            ExprKind::Break { value, .. } => {
                if let Some(value) = value {
                    self.check_expr(value)?;
                }
                Ok(self.ctx.registry.intern(Ty::Never))
            }

            ExprKind::Continue(_) => {
                Ok(self.ctx.registry.intern(Ty::Never))
            }

            ExprKind::Ref { mutable, expr: inner } => {
                let inner_ty = self.check_expr(inner)?;
                // Create a fresh lifetime inference variable for this reference
                let lifetime = self.ctx.registry.fresh_lifetime_infer();
                Ok(self.ctx.registry.intern(Ty::Ref {
                    lifetime,
                    mutable: *mutable,
                    inner: inner_ty,
                }))
            }

            ExprKind::Deref(inner) => {
                let inner_ty = self.check_expr(inner)?;
                let resolved = self.ctx.resolve(inner_ty);
                match self.ctx.registry.get(resolved) {
                    Some(Ty::Ref { inner, .. }) | Some(Ty::Ptr { inner, .. }) => Ok(inner),
                    // Box<T> derefs to T - Ty::Box variant
                    Some(Ty::Box(box_inner)) => Ok(box_inner),
                    // Box<T> derefs to T - Ty::Adt variant (backwards compat)
                    Some(Ty::Adt { name, args, .. }) if name == "Box" && !args.is_empty() => {
                        Ok(args[0])
                    }
                    // Infer type - deref produces fresh infer (let inference resolve later)
                    Some(Ty::Infer(_)) => {
                        Ok(self.ctx.registry.fresh_infer())
                    }
                    other => {
                        self.ctx.emit_error(
                            Diagnostic::error(format!(
                                "Cannot dereference non-pointer type: {}",
                                self.ctx.display_type(resolved)
                            ))
                                .with_span(expr.span),
                        );
                        Ok(self.ctx.registry.intern(Ty::Error))
                    }
                }
            }

            ExprKind::Cast { expr: inner, ty } => {
                self.check_expr(inner)?;
                Ok(self.resolve_type(ty))
            }

            ExprKind::Range { lo, hi, .. } => {
                if let Some(lo) = lo {
                    self.check_expr(lo)?;
                }
                if let Some(hi) = hi {
                    self.check_expr(hi)?;
                }
                // Range types are ADTs, but for now return a placeholder
                Ok(self.ctx.registry.fresh_infer())
            }

            ExprKind::Closure { params, body, .. } => {
                // Save old locals
                let old_locals = self.locals.clone();

                // Bind closure parameters
                let param_tys: Vec<_> = params
                    .iter()
                    .map(|(pat, ty)| {
                        let ty_id = ty
                            .as_ref()
                            .map(|t| self.resolve_type(t))
                            .unwrap_or_else(|| self.ctx.registry.fresh_infer());
                        self.bind_pattern(pat, ty_id);
                        ty_id
                    })
                    .collect();

                let body_ty = self.check_expr(body)?;

                // Restore locals
                self.locals = old_locals;

                Ok(self.ctx.registry.intern(Ty::Fn {
                    inputs: param_tys,
                    output: body_ty,
                }))
            }

            ExprKind::Await(inner) => {
                // Await requires Future trait, for now just return the inner type
                self.check_expr(inner)
            }

            ExprKind::Try(inner) => {
                let inner_ty = self.check_expr(inner)?;
                // Try requires the ? operator behavior
                // For Result<T, E>, returns T
                // For now, return a fresh inference variable
                Ok(self.ctx.registry.fresh_infer())
            }

            ExprKind::Err => Ok(self.ctx.registry.intern(Ty::Error)),
        }
    }

    fn literal_type(&self, lit: &Literal) -> TyId {
        match lit {
            Literal::Bool(_) => self.ctx.registry.intern(Ty::Bool),
            Literal::Char(_) => self.ctx.registry.intern(Ty::Char),
            Literal::Int(_, suffix) => {
                suffix
                    .map(|s| self.ctx.registry.intern(Ty::Int(s)))
                    .unwrap_or_else(|| self.ctx.registry.fresh_infer())
            }
            Literal::Uint(_, suffix) => {
                suffix
                    .map(|s| self.ctx.registry.intern(Ty::Uint(s)))
                    .unwrap_or_else(|| self.ctx.registry.fresh_infer())
            }
            Literal::Float(_, suffix) => {
                suffix
                    .map(|s| self.ctx.registry.intern(Ty::Float(s)))
                    .unwrap_or(self.ctx.registry.intern(Ty::Float(FloatType::F64)))
            }
            Literal::Str(_) => {
                let str_ty = self.ctx.registry.intern(Ty::Str);
                let static_lifetime = self.ctx.registry.static_lifetime();
                self.ctx.registry.intern(Ty::Ref {
                    lifetime: static_lifetime,
                    mutable: false,
                    inner: str_ty,
                })
            }
            Literal::ByteStr(bytes) => {
                let u8_ty = self.ctx.registry.intern(Ty::Uint(UintType::U8));
                let arr_ty = self.ctx.registry.intern(Ty::Array {
                    elem: u8_ty,
                    len: bytes.len(),
                });
                let static_lifetime = self.ctx.registry.static_lifetime();
                self.ctx.registry.intern(Ty::Ref {
                    lifetime: static_lifetime,
                    mutable: false,
                    inner: arr_ty,
                })
            }
        }
    }

    fn resolve_path(&mut self, path: &Path, span: Span) -> Result<TyId> {
        if path.segments.is_empty() {
            return Ok(self.ctx.registry.intern(Ty::Error));
        }

        // Single segment - check locals first
        if path.segments.len() == 1 {
            let name = &path.segments[0].ident;

            // Check local variables
            if let Some(&ty) = self.locals.get(name) {
                return Ok(ty);
            }

            // Check type parameters
            if let Some(&ty) = self.type_params.get(name) {
                return Ok(ty);
            }

            // Check for functions
            for (_, item) in &self.krate.items {
                if item.name == *name {
                    if let ItemKind::Function(f) = &item.kind {
                        let inputs: Vec<_> = f
                            .sig
                            .inputs
                            .iter()
                            .map(|(_, ty)| self.resolve_type(ty))
                            .collect();
                        let output = self.resolve_type(&f.sig.output);
                        return Ok(self.ctx.registry.intern(Ty::Fn { inputs, output }));
                    }
                }
            }

            // Check for enum variant (unit variant used as value)
            for (_, item) in &self.krate.items {
                if let ItemKind::Enum(e) = &item.kind {
                    for variant in &e.variants {
                        if variant.name == *name {
                            // Found enum variant
                            return Ok(self.ctx.registry.fresh_infer());
                        }
                    }
                }
            }
        }

        // Multi-segment path - could be module::item or Enum::Variant or extern::Crate
        if path.segments.len() >= 2 {
            let type_name = &path.segments[0].ident;
            let variant_name = &path.segments[1].ident;

            // Check for external crate (e.g., serde::Serialize, std::Clone)
            let resolver = global_resolver();
            if resolver.is_external_crate(type_name) {
                let item_path: Vec<&str> = path.segments[1..].iter()
                    .map(|s| s.ident.as_str())
                    .collect();

                match resolver.resolve(type_name, &item_path) {
                    ResolvedItem::Stub(stub) => {
                        match &stub.kind {
                            StubItemKind::Trait(_trait_stub) => {
                                // Return a trait type - for now use DynTrait marker
                                // In full implementation, would track trait bounds properly
                                return Ok(self.ctx.registry.intern(Ty::DynTrait {
                                    bounds: vec![stub.item_name.clone()],
                                }));
                            }
                            StubItemKind::Type(_type_stub) => {
                                // External type - create an opaque ADT
                                // Use high bit as marker for external types
                                return Ok(self.ctx.registry.intern(Ty::Adt {
                                    def_id: 0x8000_0000,
                                    name: stub.item_name.clone(),
                                    args: vec![],
                                }));
                            }
                        }
                    }
                    ResolvedItem::Rlib(_rlib_item) => {
                        // Found in rlib but not fully parsed - treat as opaque
                        return Ok(self.ctx.registry.fresh_infer());
                    }
                    ResolvedItem::NotFound => {
                        // Not found in external crate - fall through to other checks
                    }
                }
            }

            // Check for enum variant
            if let Some(enum_def) = self.ctx.registry.get_enum(type_name) {
                for variant in &enum_def.variants {
                    if &variant.name == variant_name {
                        // Create fresh type variables for each generic parameter
                        let mut type_args = Vec::new();
                        let mut subst = HashMap::new();
                        for param in &enum_def.generics.params {
                            if let GenericParam::Type { name, .. } = param {
                                let fresh = self.ctx.registry.fresh_infer();
                                type_args.push(fresh);
                                subst.insert(name.clone(), fresh);
                            }
                        }

                        if variant.fields.is_empty() {
                            // Unit variant - return the enum type directly
                            return Ok(self.ctx.registry.intern(Ty::Adt {
                                def_id: enum_def.def_id,
                                name: type_name.clone(),
                                args: type_args,
                            }));
                        } else {
                            // Tuple variant - return a constructor function type
                            let inputs: Vec<_> = variant.fields.iter()
                                .map(|(_, field_ty)| {
                                    self.ctx.registry.resolve_hir_type_with_subst(field_ty, &subst)
                                })
                                .collect();
                            let output = self.ctx.registry.intern(Ty::Adt {
                                def_id: enum_def.def_id,
                                name: type_name.clone(),
                                args: type_args,
                            });
                            return Ok(self.ctx.registry.intern(Ty::Fn { inputs, output }));
                        }
                    }
                }
            }

            // Check for imported function
            if let Some(full_path) = self.krate.imports.get(type_name) {
                // Look up the function
                for (_, item) in &self.krate.items {
                    if item.name == *full_path {
                        if let ItemKind::Function(f) = &item.kind {
                            let inputs: Vec<_> = f
                                .sig
                                .inputs
                                .iter()
                                .map(|(_, ty)| self.resolve_type(ty))
                                .collect();
                            let output = self.resolve_type(&f.sig.output);
                            return Ok(self.ctx.registry.intern(Ty::Fn { inputs, output }));
                        }
                    }
                }
            }

            // Check for Type::method (static method / associated function)
            let method_name = variant_name; // reuse the variable
            for (_, item) in &self.krate.items {
                if let ItemKind::Impl(impl_block) = &item.kind {
                    // Check if impl is for this type
                    if let TypeKind::Path(impl_path) = &impl_block.self_ty.kind {
                        if let Some(seg) = impl_path.segments.first() {
                            if &seg.ident == type_name {
                                // Found impl for this type, look for the method by DefId
                                for method_def_id in &impl_block.items {
                                    if let Some(method_item) = self.krate.items.get(method_def_id) {
                                        if &method_item.name == method_name {
                                            if let ItemKind::Function(f) = &method_item.kind {
                                                // Build the function type
                                                let inputs: Vec<_> = f
                                                    .sig
                                                    .inputs
                                                    .iter()
                                                    .map(|(_, ty)| self.resolve_type(ty))
                                                    .collect();
                                                let output = self.resolve_type(&f.sig.output);
                                                return Ok(self.ctx.registry.intern(Ty::Fn { inputs, output }));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Unknown path
        Ok(self.ctx.registry.fresh_infer())
    }

    fn check_field_access(&mut self, base_ty: TyId, field: &str, span: Span) -> Result<TyId> {
        let resolved = self.ctx.resolve(base_ty);

        // Handle references - auto-deref
        let inner_ty = match self.ctx.registry.get(resolved) {
            Some(Ty::Ref { inner, .. }) => inner,
            _ => resolved,
        };

        match self.ctx.registry.get(inner_ty) {
            Some(Ty::Adt { name, args, .. }) => {
                if let Some(struct_def) = self.ctx.registry.get_struct(&name) {
                    // Build substitution map
                    let mut subst = HashMap::new();
                    for (i, param) in struct_def.generics.params.iter().enumerate() {
                        if let GenericParam::Type { name: param_name, .. } = param {
                            if i < args.len() {
                                subst.insert(param_name.clone(), args[i]);
                            }
                        }
                    }

                    // Find field
                    for (field_name, field_ty) in &struct_def.fields {
                        if field_name == field {
                            return Ok(self.ctx.registry.resolve_hir_type_with_subst(field_ty, &subst));
                        }
                    }

                    self.ctx.emit_error(
                        Diagnostic::error(format!("No field `{}` on type `{}`", field, name))
                            .with_span(span),
                    );
                }
            }
            Some(Ty::Tuple(elems)) => {
                // Tuple field access like t.0, t.1
                if let Ok(index) = field.parse::<usize>() {
                    if index < elems.len() {
                        return Ok(elems[index]);
                    }
                }
            }
            _ => {}
        }

        Ok(self.ctx.registry.fresh_infer())
    }

    fn check_call(&mut self, func_ty: TyId, args: &[Expr], span: Span) -> Result<TyId> {
        let resolved = self.ctx.resolve(func_ty);

        match self.ctx.registry.get(resolved) {
            Some(Ty::Fn { inputs, output }) => {
                if args.len() != inputs.len() {
                    self.ctx.emit_error(
                        Diagnostic::error(format!(
                            "Expected {} arguments, got {}",
                            inputs.len(),
                            args.len()
                        ))
                        .with_span(span),
                    );
                }

                for (arg, &expected) in args.iter().zip(inputs.iter()) {
                    let arg_ty = self.check_expr(arg)?;
                    if !self.ctx.unify(expected, arg_ty) {
                        self.ctx.emit_error(
                            Diagnostic::error(format!(
                                "Expected {}, got {}",
                                self.ctx.display_type(expected),
                                self.ctx.display_type(arg_ty)
                            ))
                            .with_span(arg.span),
                        );
                    }
                }

                Ok(output)
            }
            _ => {
                // Check args anyway
                for arg in args {
                    self.check_expr(arg)?;
                }
                Ok(self.ctx.registry.fresh_infer())
            }
        }
    }

    fn check_method_call(
        &mut self,
        receiver_ty: TyId,
        method: &str,
        args: &[Expr],
        span: Span,
    ) -> Result<TyId> {
        // Try to resolve the method using the method table
        if let Some(method_info) = self.ctx.registry.resolve_method(receiver_ty, method) {
            // Look up the function definition to get its signature
            if let Some(item) = self.krate.items.get(&method_info.def_id) {
                if let crate::hir::ItemKind::Function(func) = &item.kind {
                    // Build substitution map for generics if the receiver is generic
                    let subst = self.build_receiver_subst(receiver_ty);

                    // Resolve the return type
                    let return_ty = self.ctx.registry.resolve_hir_type_with_subst(&func.sig.output, &subst);

                    // Check argument count (skip first arg which is self)
                    let expected_args = if func.sig.inputs.first().map(|(n, _)| n.as_str()) == Some("self") {
                        func.sig.inputs.len() - 1
                    } else {
                        func.sig.inputs.len()
                    };

                    if args.len() != expected_args {
                        self.ctx.emit_error(
                            Diagnostic::error(format!(
                                "Method `{}` expected {} argument(s), got {}",
                                method, expected_args, args.len()
                            ))
                            .with_span(span),
                        );
                    }

                    // Check argument types
                    let param_offset = if func.sig.inputs.first().map(|(n, _)| n.as_str()) == Some("self") { 1 } else { 0 };
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr(arg)?;
                        if let Some((_, param_ty)) = func.sig.inputs.get(i + param_offset) {
                            let expected_ty = self.ctx.registry.resolve_hir_type_with_subst(param_ty, &subst);
                            if !self.ctx.unify(expected_ty, arg_ty) {
                                self.ctx.emit_error(
                                    Diagnostic::error(format!(
                                        "Method `{}` expected {}, got {}",
                                        method,
                                        self.ctx.display_type(expected_ty),
                                        self.ctx.display_type(arg_ty)
                                    ))
                                    .with_span(arg.span),
                                );
                            }
                        }
                    }

                    return Ok(return_ty);
                }
            }
        }

        // Fallback: check args and infer return type
        for arg in args {
            self.check_expr(arg)?;
        }

        // Try built-in method signatures for common types
        let resolved_ty = self.ctx.resolve(receiver_ty);
        if let Some(ty) = self.ctx.registry.get(resolved_ty) {
            match ty {
                Ty::Vec(elem_ty) => {
                    match method {
                        "push" => return Ok(self.ctx.registry.intern(Ty::Unit)),
                        "pop" => return Ok(self.ctx.registry.intern(Ty::Option(elem_ty))),
                        "len" | "capacity" => return Ok(self.ctx.registry.intern(Ty::Uint(crate::hir::UintType::Usize))),
                        "get" => return Ok(self.ctx.registry.intern(Ty::Option(elem_ty))),
                        "is_empty" => return Ok(self.ctx.registry.intern(Ty::Bool)),
                        "sum" => return Ok(elem_ty), // Simplified
                        _ => {}
                    }
                }
                Ty::String => {
                    match method {
                        "len" | "capacity" => return Ok(self.ctx.registry.intern(Ty::Uint(crate::hir::UintType::Usize))),
                        "is_empty" => return Ok(self.ctx.registry.intern(Ty::Bool)),
                        "as_str" => return Ok(self.ctx.registry.intern(Ty::Str)),
                        "push" | "push_str" => return Ok(self.ctx.registry.intern(Ty::Unit)),
                        _ => {}
                    }
                }
                Ty::Option(inner_ty) => {
                    match method {
                        "unwrap" | "expect" => return Ok(inner_ty),
                        "is_some" | "is_none" => return Ok(self.ctx.registry.intern(Ty::Bool)),
                        "map" | "and_then" => return Ok(self.ctx.registry.fresh_infer()),
                        _ => {}
                    }
                }
                Ty::Result { ok, .. } => {
                    match method {
                        "unwrap" | "expect" => return Ok(ok),
                        "is_ok" | "is_err" => return Ok(self.ctx.registry.intern(Ty::Bool)),
                        "ok" => return Ok(self.ctx.registry.intern(Ty::Option(ok))),
                        _ => {}
                    }
                }
                Ty::Slice(elem_ty) => {
                    match method {
                        "len" => return Ok(self.ctx.registry.intern(Ty::Uint(crate::hir::UintType::Usize))),
                        "is_empty" => return Ok(self.ctx.registry.intern(Ty::Bool)),
                        "first" | "last" => return Ok(self.ctx.registry.intern(Ty::Option(elem_ty))),
                        "get" => return Ok(self.ctx.registry.intern(Ty::Option(elem_ty))),
                        "iter" => return Ok(self.ctx.registry.fresh_infer()), // Iterator type
                        _ => {}
                    }
                }
                Ty::Ref { inner, .. } => {
                    // Check if inner is a slice and delegate
                    if let Some(Ty::Slice(elem_ty)) = self.ctx.registry.get(inner) {
                        match method {
                            "len" => return Ok(self.ctx.registry.intern(Ty::Uint(crate::hir::UintType::Usize))),
                            "is_empty" => return Ok(self.ctx.registry.intern(Ty::Bool)),
                            "first" | "last" => return Ok(self.ctx.registry.intern(Ty::Option(elem_ty))),
                            "get" => return Ok(self.ctx.registry.intern(Ty::Option(elem_ty))),
                            "iter" => return Ok(self.ctx.registry.fresh_infer()),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        // Unknown method - return inference variable
        Ok(self.ctx.registry.fresh_infer())
    }

    /// Build substitution map from a generic receiver type
    fn build_receiver_subst(&self, receiver_ty: TyId) -> std::collections::HashMap<String, TyId> {
        let mut subst = std::collections::HashMap::new();

        if let Some(ty) = self.ctx.registry.get(receiver_ty) {
            match ty {
                Ty::Vec(elem) => {
                    subst.insert("T".to_string(), elem);
                }
                Ty::Box(inner) => {
                    subst.insert("T".to_string(), inner);
                }
                Ty::Option(inner) => {
                    subst.insert("T".to_string(), inner);
                }
                Ty::Result { ok, err } => {
                    subst.insert("T".to_string(), ok);
                    subst.insert("E".to_string(), err);
                }
                Ty::Adt { args, .. } => {
                    // Generic ADT - assume T, U, V naming convention
                    for (i, &arg) in args.iter().enumerate() {
                        let name = match i {
                            0 => "T",
                            1 => "U",
                            2 => "V",
                            _ => continue,
                        };
                        subst.insert(name.to_string(), arg);
                    }
                }
                _ => {}
            }
        }

        subst
    }

    fn check_struct_expr(
        &mut self,
        path: &Path,
        fields: &[(String, Expr)],
        rest: Option<&Expr>,
        span: Span,
    ) -> Result<TyId> {
        let struct_name = path.segments.last().map(|s| s.ident.as_str()).unwrap_or("");

        // Resolve type arguments
        let type_args: Vec<TyId> = path
            .segments
            .last()
            .and_then(|s| s.args.as_ref())
            .map(|args| {
                args.args
                    .iter()
                    .filter_map(|arg| match arg {
                        GenericArg::Type(ty) => Some(self.resolve_type(ty)),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();

        if let Some(struct_def) = self.ctx.registry.get_struct(struct_name) {
            // If no explicit type args, try to infer from type params in scope
            let effective_type_args: Vec<TyId> = if type_args.is_empty() && !struct_def.generics.params.is_empty() {
                // Try to use type parameters from current scope
                struct_def.generics.params.iter().filter_map(|p| {
                    if let GenericParam::Type { name, .. } = p {
                        self.type_params.get(name).copied()
                    } else {
                        None
                    }
                }).collect()
            } else {
                type_args
            };

            // Build substitution map
            let mut subst = HashMap::new();
            for (i, param) in struct_def.generics.params.iter().enumerate() {
                if let GenericParam::Type { name, .. } = param {
                    if i < effective_type_args.len() {
                        subst.insert(name.clone(), effective_type_args[i]);
                    }
                }
            }

            // Check each field
            for (field_name, field_expr) in fields {
                let field_ty = self.check_expr(field_expr)?;

                if let Some((_, expected_hir_ty)) = struct_def.fields.iter().find(|(n, _)| n == field_name) {
                    let expected_ty = self.ctx.registry.resolve_hir_type_with_subst(expected_hir_ty, &subst);
                    if !self.ctx.unify(expected_ty, field_ty) {
                        self.ctx.emit_error(
                            Diagnostic::error(format!(
                                "Field `{}`: expected {}, got {}",
                                field_name,
                                self.ctx.display_type(expected_ty),
                                self.ctx.display_type(field_ty)
                            ))
                            .with_span(field_expr.span),
                        );
                    }
                }
            }

            if let Some(rest_expr) = rest {
                self.check_expr(rest_expr)?;
            }

            return Ok(self.ctx.registry.intern(Ty::Adt {
                def_id: struct_def.def_id,
                name: struct_name.to_string(),
                args: effective_type_args,
            }));
        }

        // Check for enum variant constructor
        for (_, item) in &self.krate.items {
            if let ItemKind::Enum(e) = &item.kind {
                for variant in &e.variants {
                    if variant.name == struct_name {
                        // Found enum variant constructor
                        for (_, field_expr) in fields {
                            self.check_expr(field_expr)?;
                        }
                        return Ok(self.ctx.registry.fresh_infer());
                    }
                }
            }
        }

        Ok(self.ctx.registry.fresh_infer())
    }

    fn check_binop(&self, op: BinaryOp, lhs: TyId, rhs: TyId, span: Span) -> Result<TyId> {
        match op {
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                Ok(self.ctx.registry.intern(Ty::Bool))
            }
            BinaryOp::And | BinaryOp::Or => {
                let bool_ty = self.ctx.registry.intern(Ty::Bool);
                if !self.ctx.unify(lhs, bool_ty) || !self.ctx.unify(rhs, bool_ty) {
                    self.ctx.emit_error(
                        Diagnostic::error("Logical operators require bool operands")
                            .with_span(span),
                    );
                }
                Ok(bool_ty)
            }
            _ => {
                // Allow pointer arithmetic: ptr + int, int + ptr
                let lhs_ty = self.ctx.registry.get(self.ctx.resolve(lhs));
                let rhs_ty = self.ctx.registry.get(self.ctx.resolve(rhs));
                let is_ptr_arithmetic = matches!(
                    (&lhs_ty, &rhs_ty),
                    (Some(Ty::Ptr { .. }), Some(Ty::Int(_) | Ty::Uint(_))) |
                    (Some(Ty::Int(_) | Ty::Uint(_)), Some(Ty::Ptr { .. }))
                );

                if !is_ptr_arithmetic && !self.ctx.unify(lhs, rhs) {
                    self.ctx.emit_error(
                        Diagnostic::error(format!(
                            "Cannot apply {:?} to {} and {}",
                            op,
                            self.ctx.display_type(lhs),
                            self.ctx.display_type(rhs)
                        ))
                        .with_span(span),
                    );
                }
                // For pointer arithmetic, return pointer type
                if is_ptr_arithmetic {
                    if matches!(&lhs_ty, Some(Ty::Ptr { .. })) { Ok(lhs) } else { Ok(rhs) }
                } else {
                    Ok(lhs)
                }
            }
        }
    }

    fn check_const(&mut self, c: &Const) -> Result<()> {
        let expected = self.resolve_type(&c.ty);
        let actual = self.check_expr(&c.value)?;
        if !self.ctx.unify(expected, actual) {
            self.ctx.emit_error(Diagnostic::error(format!(
                "Const type mismatch: expected {}, found {}",
                self.ctx.display_type(expected),
                self.ctx.display_type(actual)
            )));
        }
        Ok(())
    }

    fn check_static(&mut self, s: &Static) -> Result<()> {
        let expected = self.resolve_type(&s.ty);
        let actual = self.check_expr(&s.value)?;
        if !self.ctx.unify(expected, actual) {
            self.ctx.emit_error(Diagnostic::error(format!(
                "Static type mismatch: expected {}, found {}",
                self.ctx.display_type(expected),
                self.ctx.display_type(actual)
            )));
        }
        Ok(())
    }

    fn check_impl(&mut self, impl_block: &Impl) -> Result<()> {
        // Register impl-level lifetime parameters
        for param in impl_block.generics.params.iter() {
            if let GenericParam::Lifetime { name: lt_name } = param {
                let lifetime_id = self.ctx.registry.named_lifetime(lt_name);
                self.lifetime_params.insert(lt_name.clone(), lifetime_id);
            }
        }

        // Register impl-level type parameters
        for (i, param) in impl_block.generics.params.iter().enumerate() {
            if let GenericParam::Type { name: param_name, .. } = param {
                let ty_id = self.ctx.registry.intern(Ty::Param {
                    name: param_name.clone(),
                    index: i,
                });
                self.type_params.insert(param_name.clone(), ty_id);
            }
        }

        // Check each method in the impl block
        for method_def_id in &impl_block.items {
            if let Some(method_item) = self.krate.items.get(method_def_id) {
                if let ItemKind::Function(f) = &method_item.kind {
                    self.check_function(f, &method_item.name)?;
                }
            }
        }

        self.type_params.clear();
        self.lifetime_params.clear();
        Ok(())
    }
}

/// Check a crate with parallel function checking
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
