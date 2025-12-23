//! Lowering from syn AST to HIR

use crate::error::Span;
use crate::hir::*;
use crate::parser::Parser;
use std::sync::atomic::Ordering;
use syn::spanned::Spanned;

pub struct Lowerer<'a> {
    parser: &'a Parser,
    file_id: u32,
}

impl<'a> Lowerer<'a> {
    pub fn new(parser: &'a Parser, file_id: u32) -> Self {
        Self { parser, file_id }
    }

    /// Extract derive traits from #[derive(...)] attributes
    fn extract_derives(&self, attrs: &[syn::Attribute]) -> Vec<String> {
        let mut derives = Vec::new();
        for attr in attrs {
            if attr.path().is_ident("derive") {
                if let syn::Meta::List(list) = &attr.meta {
                    // Parse the derive list tokens
                    let tokens = list.tokens.to_string();
                    for part in tokens.split(',') {
                        let trait_name = part.trim();
                        if !trait_name.is_empty() {
                            derives.push(trait_name.to_string());
                        }
                    }
                }
            }
        }
        derives
    }

    fn span(&self, _s: proc_macro2::Span) -> Span {
        // Use dummy span - proc_macro2 span position requires special features
        Span::new(self.file_id, 0, 0)
    }

    fn alloc_def_id(&self) -> DefId {
        self.parser.next_def_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn lower_file(&self, file: &syn::File, krate: &mut Crate) {
        self.lower_items(&file.items, krate, "");
    }

    fn lower_items(&self, items: &[syn::Item], krate: &mut Crate, prefix: &str) {
        for item in items {
            // Process use statements to build import aliases
            if let syn::Item::Use(u) = item {
                self.process_use(&u.tree, "", krate, prefix);
                continue;
            }

            if let Some(mut hir_item) = self.lower_item(item) {
                // Prefix item names in modules (except main at top level)
                if !prefix.is_empty() && !matches!(&hir_item.kind, ItemKind::Module(_)) {
                    // For functions and structs inside modules, mangle the name
                    // This allows them to be found via module::item paths
                    hir_item.name = format!("{}::{}", prefix, hir_item.name);
                }

                let id = hir_item.id;
                if let ItemKind::Function(ref f) = hir_item.kind {
                    // Only look for main at the top level with original name
                    if prefix.is_empty() && hir_item.name == "main" && f.sig.inputs.is_empty() {
                        krate.entry_point = Some(id);
                    }
                }
                krate.items.insert(id, hir_item);
            }

            // Also lower methods from impl blocks and update impl.items with correct DefIds
            if let syn::Item::Impl(i) = item {
                let type_name = self.type_name(&i.self_ty);
                let full_type = if prefix.is_empty() {
                    type_name.clone()
                } else {
                    format!("{}::{}", prefix, type_name)
                };

                // Collect the actual method DefIds as we lower them
                let mut method_def_ids = Vec::new();
                let impl_generics = &i.generics;
                for impl_item in &i.items {
                    if let syn::ImplItem::Fn(f) = impl_item {
                        let method = self.lower_impl_fn(f, &full_type, impl_generics);
                        method_def_ids.push(method.id);
                        krate.items.insert(method.id, method);
                    }
                }

                // Update the impl block's items with the correct DefIds
                // Find the impl we just inserted and fix its items
                for (_, hir_item) in krate.items.iter_mut() {
                    if let ItemKind::Impl(impl_block) = &mut hir_item.kind {
                        if let TypeKind::Path(p) = &impl_block.self_ty.kind {
                            if p.segments.first().map(|s| &s.ident) == Some(&type_name) {
                                impl_block.items = method_def_ids.clone();
                                break;
                            }
                        }
                    }
                }
            }

            // Recursively lower inline modules
            if let syn::Item::Mod(m) = item {
                if let Some((_, content_items)) = &m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    self.lower_items(content_items, krate, &mod_prefix);
                }
            }

            // Process derive macros for structs
            if let syn::Item::Struct(s) = item {
                let derives = self.extract_derives(&s.attrs);
                let type_name = s.ident.to_string();
                let full_type = if prefix.is_empty() {
                    type_name.clone()
                } else {
                    format!("{}::{}", prefix, type_name)
                };

                for derive in derives {
                    if let Some(method) = self.generate_derive_method(&derive, &full_type, s) {
                        krate.items.insert(method.id, method);
                    }
                }
            }

            // Process derive macros for enums
            if let syn::Item::Enum(e) = item {
                let derives = self.extract_derives(&e.attrs);
                let type_name = e.ident.to_string();
                let full_type = if prefix.is_empty() {
                    type_name.clone()
                } else {
                    format!("{}::{}", prefix, type_name)
                };

                for derive in derives {
                    if let Some(method) = self.generate_derive_method_enum(&derive, &full_type, e) {
                        krate.items.insert(method.id, method);
                    }
                }
            }
        }
    }

    /// Process a use tree to extract import aliases
    fn process_use(&self, tree: &syn::UseTree, path_prefix: &str, krate: &mut Crate, module_prefix: &str) {
        match tree {
            syn::UseTree::Path(p) => {
                // use foo::bar::... - accumulate path
                let new_prefix = if path_prefix.is_empty() {
                    p.ident.to_string()
                } else {
                    format!("{}::{}", path_prefix, p.ident)
                };
                self.process_use(&p.tree, &new_prefix, krate, module_prefix);
            }
            syn::UseTree::Name(n) => {
                // use foo::bar; - final name
                let full_path = if path_prefix.is_empty() {
                    n.ident.to_string()
                } else {
                    format!("{}::{}", path_prefix, n.ident)
                };
                let alias = n.ident.to_string();
                // If we're in a module, the import is local to that module
                let key = if module_prefix.is_empty() {
                    alias
                } else {
                    format!("{}::{}", module_prefix, alias)
                };
                krate.imports.insert(key, full_path);
            }
            syn::UseTree::Rename(r) => {
                // use foo::bar as baz;
                let full_path = if path_prefix.is_empty() {
                    r.ident.to_string()
                } else {
                    format!("{}::{}", path_prefix, r.ident)
                };
                let alias = r.rename.to_string();
                let key = if module_prefix.is_empty() {
                    alias
                } else {
                    format!("{}::{}", module_prefix, alias)
                };
                krate.imports.insert(key, full_path);
            }
            syn::UseTree::Group(g) => {
                // use foo::{bar, baz};
                for tree in &g.items {
                    self.process_use(tree, path_prefix, krate, module_prefix);
                }
            }
            syn::UseTree::Glob(_) => {
                // use foo::*; - not supported yet, would need to enumerate all items
            }
        }
    }
    
    fn type_name(&self, ty: &syn::Type) -> String {
        match ty {
            syn::Type::Path(tp) => {
                tp.path.segments.last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default()
            }
            _ => String::new(),
        }
    }
    
    fn lower_impl_fn(&self, f: &syn::ImplItemFn, type_name: &str, impl_generics: &syn::Generics) -> Item {
        let id = self.alloc_def_id();
        let method_name = f.sig.ident.to_string();
        let full_name = format!("{}_{}", type_name, method_name);

        let mut inputs: Vec<(String, Type)> = Vec::new();

        // Build generic args from impl generics for self type
        let self_type_args: Option<GenericArgs> = if impl_generics.params.is_empty() {
            None
        } else {
            let args: Vec<GenericArg> = impl_generics.params.iter().filter_map(|p| {
                match p {
                    syn::GenericParam::Type(tp) => {
                        let name = tp.ident.to_string();
                        Some(GenericArg::Type(Type {
                            kind: TypeKind::Path(Path {
                                segments: vec![PathSegment {
                                    ident: name,
                                    args: None,
                                }],
                            }),
                            span: Span::dummy(),
                        }))
                    }
                    _ => None,
                }
            }).collect();
            if args.is_empty() { None } else { Some(GenericArgs { args }) }
        };

        for arg in &f.sig.inputs {
            match arg {
                syn::FnArg::Receiver(_r) => {
                    // self parameter - becomes pointer to struct with type args
                    inputs.push(("self".to_string(), Type {
                        kind: TypeKind::Path(Path {
                            segments: vec![PathSegment {
                                ident: type_name.to_string(),
                                args: self_type_args.clone(),
                            }],
                        }),
                        span: Span::dummy(),
                    }));
                }
                syn::FnArg::Typed(pat_ty) => {
                    let name = if let syn::Pat::Ident(pi) = &*pat_ty.pat {
                        pi.ident.to_string()
                    } else {
                        "_".to_string()
                    };
                    inputs.push((name, self.lower_type(&pat_ty.ty)));
                }
            }
        }
        
        let output = match &f.sig.output {
            syn::ReturnType::Default => Type {
                kind: TypeKind::Unit,
                span: Span::dummy(),
            },
            syn::ReturnType::Type(_, ty) => self.lower_type(ty),
        };
        
        let sig = FnSig {
            inputs,
            output,
            generics: self.lower_generics(&f.sig.generics),
        };
        
        let body = Some(self.lower_block(&f.block));
        
        Item {
            id,
            name: full_name,
            kind: ItemKind::Function(Function {
                sig,
                body,
                is_async: f.sig.asyncness.is_some(),
                is_const: f.sig.constness.is_some(),
                is_unsafe: f.sig.unsafety.is_some(),
            }),
            visibility: self.lower_visibility(&f.vis),
            span: self.span(f.span()),
        }
    }

    /// Generate a method for a derive macro on a struct
    fn generate_derive_method(&self, derive: &str, type_name: &str, s: &syn::ItemStruct) -> Option<Item> {
        match derive {
            "Clone" => self.generate_clone_method(type_name, s),
            "Copy" => None,  // Copy is a marker trait, no method needed
            "Debug" => self.generate_debug_method(type_name, s),
            "Default" => self.generate_default_method(type_name, s),
            _ => None,
        }
    }

    /// Generate a method for a derive macro on an enum
    fn generate_derive_method_enum(&self, derive: &str, type_name: &str, _e: &syn::ItemEnum) -> Option<Item> {
        match derive {
            "Clone" | "Copy" | "Debug" => None,  // Enums handled by existing enum machinery
            _ => None,
        }
    }

    /// Generate Clone::clone method for a struct
    fn generate_clone_method(&self, type_name: &str, s: &syn::ItemStruct) -> Option<Item> {
        let id = self.alloc_def_id();
        let full_name = format!("{}_clone", type_name);

        // Clone takes &self and returns Self
        let self_ty = Type {
            kind: TypeKind::Path(Path {
                segments: vec![PathSegment { ident: type_name.to_string(), args: None }],
            }),
            span: Span::new(self.file_id, 0, 0),
        };

        let sig = FnSig {
            inputs: vec![("self".to_string(), self_ty.clone())],
            output: self_ty.clone(),
            generics: Generics::default(),
        };

        // Body: return self (since we treat everything as Copy for now)
        let body = Block {
            stmts: Vec::new(),
            expr: Some(Box::new(Expr::unhashed(
                ExprKind::Path(Path {
                    segments: vec![PathSegment { ident: "self".to_string(), args: None }],
                }),
                Span::new(self.file_id, 0, 0),
            ))),
            span: Span::new(self.file_id, 0, 0),
        };

        Some(Item {
            id,
            name: full_name,
            kind: ItemKind::Function(Function {
                sig,
                body: Some(body),
                is_async: false,
                is_const: false,
                is_unsafe: false,
            }),
            visibility: Visibility::Public,
            span: Span::new(self.file_id, 0, 0),
        })
    }

    /// Generate Debug method (prints type name and fields)
    fn generate_debug_method(&self, type_name: &str, _s: &syn::ItemStruct) -> Option<Item> {
        // For now, Debug just prints the type name
        // A full implementation would iterate fields
        let id = self.alloc_def_id();
        let full_name = format!("{}_debug", type_name);

        let self_ty = Type {
            kind: TypeKind::Path(Path {
                segments: vec![PathSegment { ident: type_name.to_string(), args: None }],
            }),
            span: Span::new(self.file_id, 0, 0),
        };

        let sig = FnSig {
            inputs: vec![("self".to_string(), self_ty)],
            output: Type { kind: TypeKind::Unit, span: Span::new(self.file_id, 0, 0) },
            generics: Generics::default(),
        };

        // Body: call bolt_print_str with type name
        let body = Block {
            stmts: Vec::new(),
            expr: Some(Box::new(Expr::unhashed(
                ExprKind::Lit(Literal::Int(0, None)),  // Placeholder
                Span::new(self.file_id, 0, 0),
            ))),
            span: Span::new(self.file_id, 0, 0),
        };

        Some(Item {
            id,
            name: full_name,
            kind: ItemKind::Function(Function {
                sig,
                body: Some(body),
                is_async: false,
                is_const: false,
                is_unsafe: false,
            }),
            visibility: Visibility::Public,
            span: Span::new(self.file_id, 0, 0),
        })
    }

    /// Generate Default::default method
    fn generate_default_method(&self, type_name: &str, s: &syn::ItemStruct) -> Option<Item> {
        let id = self.alloc_def_id();
        let full_name = format!("{}_default", type_name);

        let self_ty = Type {
            kind: TypeKind::Path(Path {
                segments: vec![PathSegment { ident: type_name.to_string(), args: None }],
            }),
            span: Span::new(self.file_id, 0, 0),
        };

        let sig = FnSig {
            inputs: Vec::new(),  // Default takes no args
            output: self_ty,
            generics: Generics::default(),
        };

        // Body: construct struct with default values (0 for all fields)
        let fields: Vec<(String, Expr)> = match &s.fields {
            syn::Fields::Named(named) => {
                named.named.iter().map(|f| {
                    let name = f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default();
                    let expr = Expr::unhashed(
                        ExprKind::Lit(Literal::Int(0, None)),
                        Span::new(self.file_id, 0, 0),
                    );
                    (name, expr)
                }).collect()
            }
            _ => Vec::new(),
        };

        let body = Block {
            stmts: Vec::new(),
            expr: Some(Box::new(Expr::unhashed(
                ExprKind::Struct {
                    path: Path {
                        segments: vec![PathSegment { ident: type_name.to_string(), args: None }],
                    },
                    fields,
                    rest: None,
                },
                Span::new(self.file_id, 0, 0),
            ))),
            span: Span::new(self.file_id, 0, 0),
        };

        Some(Item {
            id,
            name: full_name,
            kind: ItemKind::Function(Function {
                sig,
                body: Some(body),
                is_async: false,
                is_const: false,
                is_unsafe: false,
            }),
            visibility: Visibility::Public,
            span: Span::new(self.file_id, 0, 0),
        })
    }

    fn lower_item(&self, item: &syn::Item) -> Option<Item> {
        match item {
            syn::Item::Fn(f) => Some(self.lower_fn(f)),
            syn::Item::Struct(s) => Some(self.lower_struct(s)),
            syn::Item::Enum(e) => Some(self.lower_enum(e)),
            syn::Item::Const(c) => Some(self.lower_const(c)),
            syn::Item::Static(s) => Some(self.lower_static(s)),
            syn::Item::Impl(i) => Some(self.lower_impl(i)),
            syn::Item::Trait(t) => Some(self.lower_trait(t)),
            syn::Item::Type(t) => Some(self.lower_type_alias(t)),
            syn::Item::Mod(m) => Some(self.lower_mod(m)),
            syn::Item::Use(_) => None,
            syn::Item::ExternCrate(_) => None,
            _ => None,
        }
    }

    fn lower_fn(&self, f: &syn::ItemFn) -> Item {
        let id = self.alloc_def_id();
        let name = f.sig.ident.to_string();

        let inputs = f
            .sig
            .inputs
            .iter()
            .filter_map(|arg| {
                if let syn::FnArg::Typed(pat_ty) = arg {
                    let name = if let syn::Pat::Ident(pi) = &*pat_ty.pat {
                        pi.ident.to_string()
                    } else {
                        "_".to_string()
                    };
                    Some((name, self.lower_type(&pat_ty.ty)))
                } else {
                    None
                }
            })
            .collect();

        let output = match &f.sig.output {
            syn::ReturnType::Default => Type {
                kind: TypeKind::Unit,
                span: Span::dummy(),
            },
            syn::ReturnType::Type(_, ty) => self.lower_type(ty),
        };

        let sig = FnSig {
            inputs,
            output,
            generics: self.lower_generics(&f.sig.generics),
        };

        let body = Some(self.lower_block(&f.block));

        Item {
            id,
            name,
            kind: ItemKind::Function(Function {
                sig,
                body,
                is_async: f.sig.asyncness.is_some(),
                is_const: f.sig.constness.is_some(),
                is_unsafe: f.sig.unsafety.is_some(),
            }),
            visibility: self.lower_visibility(&f.vis),
            span: self.span(f.span()),
        }
    }

    fn lower_struct(&self, s: &syn::ItemStruct) -> Item {
        let id = self.alloc_def_id();
        let name = s.ident.to_string();

        let kind = match &s.fields {
            syn::Fields::Unit => StructKind::Unit,
            syn::Fields::Unnamed(fields) => {
                StructKind::Tuple(fields.unnamed.iter().map(|f| self.lower_type(&f.ty)).collect())
            }
            syn::Fields::Named(fields) => StructKind::Named(
                fields
                    .named
                    .iter()
                    .map(|f| Field {
                        name: f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default(),
                        ty: self.lower_type(&f.ty),
                        visibility: self.lower_visibility(&f.vis),
                    })
                    .collect(),
            ),
        };

        Item {
            id,
            name,
            kind: ItemKind::Struct(Struct {
                generics: self.lower_generics(&s.generics),
                kind,
            }),
            visibility: self.lower_visibility(&s.vis),
            span: self.span(s.span()),
        }
    }

    fn lower_enum(&self, e: &syn::ItemEnum) -> Item {
        let id = self.alloc_def_id();
        let name = e.ident.to_string();

        let variants = e
            .variants
            .iter()
            .map(|v| {
                let kind = match &v.fields {
                    syn::Fields::Unit => StructKind::Unit,
                    syn::Fields::Unnamed(fields) => {
                        StructKind::Tuple(fields.unnamed.iter().map(|f| self.lower_type(&f.ty)).collect())
                    }
                    syn::Fields::Named(fields) => StructKind::Named(
                        fields
                            .named
                            .iter()
                            .map(|f| Field {
                                name: f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default(),
                                ty: self.lower_type(&f.ty),
                                visibility: Visibility::Public,
                            })
                            .collect(),
                    ),
                };

                Variant {
                    name: v.ident.to_string(),
                    kind,
                    discriminant: v.discriminant.as_ref().map(|(_, expr)| self.lower_expr(expr)),
                }
            })
            .collect();

        Item {
            id,
            name,
            kind: ItemKind::Enum(Enum {
                generics: self.lower_generics(&e.generics),
                variants,
            }),
            visibility: self.lower_visibility(&e.vis),
            span: self.span(e.span()),
        }
    }

    fn lower_const(&self, c: &syn::ItemConst) -> Item {
        let id = self.alloc_def_id();
        Item {
            id,
            name: c.ident.to_string(),
            kind: ItemKind::Const(Const {
                ty: self.lower_type(&c.ty),
                value: self.lower_expr(&c.expr),
            }),
            visibility: self.lower_visibility(&c.vis),
            span: self.span(c.span()),
        }
    }

    fn lower_static(&self, s: &syn::ItemStatic) -> Item {
        let id = self.alloc_def_id();
        Item {
            id,
            name: s.ident.to_string(),
            kind: ItemKind::Static(Static {
                ty: self.lower_type(&s.ty),
                value: self.lower_expr(&s.expr),
                mutable: matches!(s.mutability, syn::StaticMutability::Mut(_)),
            }),
            visibility: self.lower_visibility(&s.vis),
            span: self.span(s.span()),
        }
    }

    fn lower_impl(&self, i: &syn::ItemImpl) -> Item {
        let id = self.alloc_def_id();

        let trait_ref = i.trait_.as_ref().map(|(_, path, _)| self.lower_path(path));
        let self_ty = self.lower_type(&i.self_ty);

        let items = i
            .items
            .iter()
            .filter_map(|item| match item {
                syn::ImplItem::Fn(f) => Some(self.alloc_def_id()),
                _ => None,
            })
            .collect();

        Item {
            id,
            name: String::new(),
            kind: ItemKind::Impl(Impl {
                generics: self.lower_generics(&i.generics),
                trait_ref,
                self_ty,
                items,
            }),
            visibility: Visibility::Private,
            span: self.span(i.span()),
        }
    }

    fn lower_trait(&self, t: &syn::ItemTrait) -> Item {
        let id = self.alloc_def_id();
        Item {
            id,
            name: t.ident.to_string(),
            kind: ItemKind::Trait(Trait {
                generics: self.lower_generics(&t.generics),
                bounds: t.supertraits.iter().filter_map(|b| self.lower_type_param_bound(b)).collect(),
                items: Vec::new(),
            }),
            visibility: self.lower_visibility(&t.vis),
            span: self.span(t.span()),
        }
    }

    fn lower_type_alias(&self, t: &syn::ItemType) -> Item {
        let id = self.alloc_def_id();
        Item {
            id,
            name: t.ident.to_string(),
            kind: ItemKind::TypeAlias(TypeAlias {
                generics: self.lower_generics(&t.generics),
                ty: self.lower_type(&t.ty),
            }),
            visibility: self.lower_visibility(&t.vis),
            span: self.span(t.span()),
        }
    }

    fn lower_mod(&self, m: &syn::ItemMod) -> Item {
        let id = self.alloc_def_id();
        Item {
            id,
            name: m.ident.to_string(),
            kind: ItemKind::Module(Module { items: Vec::new() }),
            visibility: self.lower_visibility(&m.vis),
            span: self.span(m.span()),
        }
    }

    fn lower_generics(&self, g: &syn::Generics) -> Generics {
        let params = g
            .params
            .iter()
            .map(|p| match p {
                syn::GenericParam::Type(tp) => GenericParam::Type {
                    name: tp.ident.to_string(),
                    bounds: tp.bounds.iter().filter_map(|b| self.lower_type_param_bound(b)).collect(),
                },
                syn::GenericParam::Lifetime(lp) => GenericParam::Lifetime {
                    name: lp.lifetime.ident.to_string(),
                },
                syn::GenericParam::Const(cp) => GenericParam::Const {
                    name: cp.ident.to_string(),
                    ty: self.lower_type(&cp.ty),
                },
            })
            .collect();

        Generics {
            params,
            where_clause: Vec::new(),
        }
    }

    fn lower_type_param_bound(&self, bound: &syn::TypeParamBound) -> Option<TypeBound> {
        match bound {
            syn::TypeParamBound::Trait(tb) => Some(TypeBound::Trait(self.lower_path(&tb.path))),
            syn::TypeParamBound::Lifetime(lt) => Some(TypeBound::Lifetime(lt.ident.to_string())),
            _ => None,
        }
    }

    fn lower_visibility(&self, vis: &syn::Visibility) -> Visibility {
        match vis {
            syn::Visibility::Public(_) => Visibility::Public,
            syn::Visibility::Restricted(r) => {
                if r.path.is_ident("crate") {
                    Visibility::Crate
                } else if r.path.is_ident("super") {
                    Visibility::Super
                } else {
                    Visibility::Private
                }
            }
            syn::Visibility::Inherited => Visibility::Private,
        }
    }

    fn lower_type(&self, ty: &syn::Type) -> Type {
        let kind = match ty {
            syn::Type::Path(tp) => {
                if tp.qself.is_none() && tp.path.segments.len() == 1 {
                    let seg = &tp.path.segments[0];
                    let ident = seg.ident.to_string();
                    match ident.as_str() {
                        "bool" => TypeKind::Bool,
                        "char" => TypeKind::Char,
                        "str" => TypeKind::Str,
                        "i8" => TypeKind::Int(IntType::I8),
                        "i16" => TypeKind::Int(IntType::I16),
                        "i32" => TypeKind::Int(IntType::I32),
                        "i64" => TypeKind::Int(IntType::I64),
                        "i128" => TypeKind::Int(IntType::I128),
                        "isize" => TypeKind::Int(IntType::Isize),
                        "u8" => TypeKind::Uint(UintType::U8),
                        "u16" => TypeKind::Uint(UintType::U16),
                        "u32" => TypeKind::Uint(UintType::U32),
                        "u64" => TypeKind::Uint(UintType::U64),
                        "u128" => TypeKind::Uint(UintType::U128),
                        "usize" => TypeKind::Uint(UintType::Usize),
                        "f32" => TypeKind::Float(FloatType::F32),
                        "f64" => TypeKind::Float(FloatType::F64),
                        _ => TypeKind::Path(self.lower_path(&tp.path)),
                    }
                } else {
                    TypeKind::Path(self.lower_path(&tp.path))
                }
            }
            syn::Type::Reference(r) => TypeKind::Ref {
                lifetime: r.lifetime.as_ref().map(|lt| lt.ident.to_string()),
                mutable: r.mutability.is_some(),
                inner: Box::new(self.lower_type(&r.elem)),
            },
            syn::Type::Ptr(p) => TypeKind::Ptr {
                mutable: p.mutability.is_some(),
                inner: Box::new(self.lower_type(&p.elem)),
            },
            syn::Type::Slice(s) => TypeKind::Slice(Box::new(self.lower_type(&s.elem))),
            syn::Type::Array(a) => TypeKind::Array {
                elem: Box::new(self.lower_type(&a.elem)),
                len: 0,
            },
            syn::Type::Tuple(t) => {
                if t.elems.is_empty() {
                    TypeKind::Unit
                } else {
                    TypeKind::Tuple(t.elems.iter().map(|e| self.lower_type(e)).collect())
                }
            }
            syn::Type::Never(_) => TypeKind::Never,
            syn::Type::Infer(_) => TypeKind::Infer,
            _ => TypeKind::Error,
        };

        Type {
            kind,
            span: self.span(ty.span()),
        }
    }

    fn lower_path(&self, path: &syn::Path) -> Path {
        Path {
            segments: path
                .segments
                .iter()
                .map(|seg| PathSegment {
                    ident: seg.ident.to_string(),
                    args: match &seg.arguments {
                        syn::PathArguments::None => None,
                        syn::PathArguments::AngleBracketed(args) => Some(GenericArgs {
                            args: args
                                .args
                                .iter()
                                .filter_map(|arg| match arg {
                                    syn::GenericArgument::Type(ty) => {
                                        Some(GenericArg::Type(self.lower_type(ty)))
                                    }
                                    syn::GenericArgument::Lifetime(lt) => {
                                        Some(GenericArg::Lifetime(lt.ident.to_string()))
                                    }
                                    _ => None,
                                })
                                .collect(),
                        }),
                        syn::PathArguments::Parenthesized(_) => None,
                    },
                })
                .collect(),
        }
    }

    fn lower_block(&self, block: &syn::Block) -> Block {
        let mut stmts = Vec::new();
        let mut final_expr = None;

        for (i, stmt) in block.stmts.iter().enumerate() {
            let is_last = i == block.stmts.len() - 1;

            match stmt {
                syn::Stmt::Local(local) => {
                    // Handle type annotation in pattern (let a: f64 = 3.5 creates Pat::Type)
                    let (pattern, ty) = if let syn::Pat::Type(pt) = &local.pat {
                        (self.lower_pattern(&pt.pat), Some(self.lower_type(&pt.ty)))
                    } else {
                        (self.lower_pattern(&local.pat), None)
                    };
                    stmts.push(Stmt {
                        kind: StmtKind::Let {
                            pattern,
                            ty,
                            init: local.init.as_ref().map(|init| self.lower_expr(&init.expr)),
                        },
                        span: self.span(local.span()),
                    });
                }
                syn::Stmt::Item(_) => {}
                syn::Stmt::Expr(expr, semi) => {
                    if is_last && semi.is_none() {
                        final_expr = Some(Box::new(self.lower_expr(expr)));
                    } else if semi.is_some() {
                        stmts.push(Stmt {
                            kind: StmtKind::Semi(self.lower_expr(expr)),
                            span: self.span(expr.span()),
                        });
                    } else {
                        stmts.push(Stmt {
                            kind: StmtKind::Expr(self.lower_expr(expr)),
                            span: self.span(expr.span()),
                        });
                    }
                }
                syn::Stmt::Macro(m) => {
                    stmts.push(Stmt {
                        kind: StmtKind::Expr(Expr::unhashed(
                            ExprKind::Err,
                            self.span(m.span()),
                        )),
                        span: self.span(m.span()),
                    });
                }
            }
        }

        Block {
            stmts,
            expr: final_expr,
            span: self.span(block.span()),
        }
    }

    fn lower_pattern(&self, pat: &syn::Pat) -> Pattern {
        let kind = match pat {
            syn::Pat::Wild(_) => PatternKind::Wild,
            syn::Pat::Ident(pi) => PatternKind::Ident {
                mutable: pi.mutability.is_some(),
                name: pi.ident.to_string(),
                binding: pi.subpat.as_ref().map(|(_, p)| Box::new(self.lower_pattern(p))),
            },
            syn::Pat::Tuple(t) => {
                PatternKind::Tuple(t.elems.iter().map(|p| self.lower_pattern(p)).collect())
            }
            syn::Pat::TupleStruct(ts) => PatternKind::TupleStruct {
                path: self.lower_path(&ts.path),
                elems: ts.elems.iter().map(|p| self.lower_pattern(p)).collect(),
            },
            syn::Pat::Struct(s) => PatternKind::Struct {
                path: self.lower_path(&s.path),
                fields: s
                    .fields
                    .iter()
                    .map(|fp| FieldPattern {
                        name: fp.member.to_token_stream().to_string(),
                        pattern: self.lower_pattern(&fp.pat),
                    })
                    .collect(),
                rest: s.rest.is_some(),
            },
            syn::Pat::Path(pp) => PatternKind::Path(self.lower_path(&pp.path)),
            syn::Pat::Lit(lit) => PatternKind::Lit(self.lower_lit(&lit.lit)),
            syn::Pat::Or(or) => {
                PatternKind::Or(or.cases.iter().map(|p| self.lower_pattern(p)).collect())
            }
            syn::Pat::Reference(r) => PatternKind::Ref {
                mutable: r.mutability.is_some(),
                pattern: Box::new(self.lower_pattern(&r.pat)),
            },
            _ => PatternKind::Wild,
        };

        Pattern {
            kind,
            span: self.span(pat.span()),
        }
    }

    fn lower_expr(&self, expr: &syn::Expr) -> Expr {
        let span = self.span(expr.span());
        let kind = match expr {
            syn::Expr::Lit(lit) => ExprKind::Lit(self.lower_lit(&lit.lit)),
            syn::Expr::Path(p) => ExprKind::Path(self.lower_path(&p.path)),
            syn::Expr::Unary(u) => ExprKind::Unary {
                op: match u.op {
                    syn::UnOp::Not(_) => UnaryOp::Not,
                    syn::UnOp::Neg(_) => UnaryOp::Neg,
                    syn::UnOp::Deref(_) => {
                        return Expr::unhashed(
                            ExprKind::Deref(Box::new(self.lower_expr(&u.expr))),
                            span,
                        );
                    }
                    _ => UnaryOp::Not,
                },
                expr: Box::new(self.lower_expr(&u.expr)),
            },
            syn::Expr::Binary(b) => ExprKind::Binary {
                op: self.lower_binop(&b.op),
                lhs: Box::new(self.lower_expr(&b.left)),
                rhs: Box::new(self.lower_expr(&b.right)),
            },
            syn::Expr::Assign(a) => ExprKind::Assign {
                lhs: Box::new(self.lower_expr(&a.left)),
                rhs: Box::new(self.lower_expr(&a.right)),
            },
            syn::Expr::Index(i) => ExprKind::Index {
                expr: Box::new(self.lower_expr(&i.expr)),
                index: Box::new(self.lower_expr(&i.index)),
            },
            syn::Expr::Field(f) => ExprKind::Field {
                expr: Box::new(self.lower_expr(&f.base)),
                field: f.member.to_token_stream().to_string(),
            },
            syn::Expr::Call(c) => ExprKind::Call {
                func: Box::new(self.lower_expr(&c.func)),
                args: c.args.iter().map(|a| self.lower_expr(a)).collect(),
            },
            syn::Expr::MethodCall(m) => ExprKind::MethodCall {
                receiver: Box::new(self.lower_expr(&m.receiver)),
                method: m.method.to_string(),
                args: m.args.iter().map(|a| self.lower_expr(a)).collect(),
            },
            syn::Expr::Tuple(t) => {
                ExprKind::Tuple(t.elems.iter().map(|e| self.lower_expr(e)).collect())
            }
            syn::Expr::Array(a) => {
                ExprKind::Array(a.elems.iter().map(|e| self.lower_expr(e)).collect())
            }
            syn::Expr::If(i) => {
                // Check if this is an if-let
                if let syn::Expr::Let(let_expr) = &*i.cond {
                    ExprKind::IfLet {
                        pattern: self.lower_pattern(&let_expr.pat),
                        expr: Box::new(self.lower_expr(&let_expr.expr)),
                        then_branch: Box::new(self.lower_block(&i.then_branch)),
                        else_branch: i.else_branch.as_ref().map(|(_, e)| Box::new(self.lower_expr(e))),
                    }
                } else {
                    ExprKind::If {
                        cond: Box::new(self.lower_expr(&i.cond)),
                        then_branch: Box::new(self.lower_block(&i.then_branch)),
                        else_branch: i.else_branch.as_ref().map(|(_, e)| Box::new(self.lower_expr(e))),
                    }
                }
            }
            syn::Expr::Match(m) => ExprKind::Match {
                expr: Box::new(self.lower_expr(&m.expr)),
                arms: m
                    .arms
                    .iter()
                    .map(|arm| Arm {
                        pattern: self.lower_pattern(&arm.pat),
                        guard: arm.guard.as_ref().map(|(_, e)| Box::new(self.lower_expr(e))),
                        body: Box::new(self.lower_expr(&arm.body)),
                    })
                    .collect(),
            },
            syn::Expr::Loop(l) => ExprKind::Loop {
                body: Box::new(self.lower_block(&l.body)),
                label: l.label.as_ref().map(|l| l.name.ident.to_string()),
            },
            syn::Expr::While(w) => {
                // Check if this is a while-let
                if let syn::Expr::Let(let_expr) = &*w.cond {
                    ExprKind::WhileLet {
                        pattern: self.lower_pattern(&let_expr.pat),
                        expr: Box::new(self.lower_expr(&let_expr.expr)),
                        body: Box::new(self.lower_block(&w.body)),
                        label: w.label.as_ref().map(|l| l.name.ident.to_string()),
                    }
                } else {
                    ExprKind::While {
                        cond: Box::new(self.lower_expr(&w.cond)),
                        body: Box::new(self.lower_block(&w.body)),
                        label: w.label.as_ref().map(|l| l.name.ident.to_string()),
                    }
                }
            }
            syn::Expr::ForLoop(f) => ExprKind::For {
                pattern: self.lower_pattern(&f.pat),
                iter: Box::new(self.lower_expr(&f.expr)),
                body: Box::new(self.lower_block(&f.body)),
                label: f.label.as_ref().map(|l| l.name.ident.to_string()),
            },
            syn::Expr::Block(b) => ExprKind::Block(Box::new(self.lower_block(&b.block))),
            syn::Expr::Return(r) => {
                ExprKind::Return(r.expr.as_ref().map(|e| Box::new(self.lower_expr(e))))
            }
            syn::Expr::Break(b) => ExprKind::Break {
                label: b.label.as_ref().map(|l| l.ident.to_string()),
                value: b.expr.as_ref().map(|e| Box::new(self.lower_expr(e))),
            },
            syn::Expr::Continue(c) => ExprKind::Continue(c.label.as_ref().map(|l| l.ident.to_string())),
            syn::Expr::Reference(r) => ExprKind::Ref {
                mutable: r.mutability.is_some(),
                expr: Box::new(self.lower_expr(&r.expr)),
            },
            syn::Expr::Cast(c) => ExprKind::Cast {
                expr: Box::new(self.lower_expr(&c.expr)),
                ty: self.lower_type(&c.ty),
            },
            syn::Expr::Closure(c) => ExprKind::Closure {
                params: c
                    .inputs
                    .iter()
                    .map(|p| (self.lower_pattern(p), None))
                    .collect(),
                body: Box::new(self.lower_expr(&c.body)),
                is_async: c.asyncness.is_some(),
                is_move: c.capture.is_some(),
            },
            syn::Expr::Await(a) => ExprKind::Await(Box::new(self.lower_expr(&a.base))),
            syn::Expr::Try(t) => ExprKind::Try(Box::new(self.lower_expr(&t.expr))),
            syn::Expr::Paren(p) => return self.lower_expr(&p.expr),
            syn::Expr::Struct(s) => ExprKind::Struct {
                path: self.lower_path(&s.path),
                fields: s.fields.iter().map(|f| {
                    let name = match &f.member {
                        syn::Member::Named(ident) => ident.to_string(),
                        syn::Member::Unnamed(idx) => idx.index.to_string(),
                    };
                    (name, self.lower_expr(&f.expr))
                }).collect(),
                rest: s.rest.as_ref().map(|r| Box::new(self.lower_expr(r))),
            },
            syn::Expr::Range(r) => ExprKind::Range {
                lo: r.start.as_ref().map(|e| Box::new(self.lower_expr(e))),
                hi: r.end.as_ref().map(|e| Box::new(self.lower_expr(e))),
                inclusive: matches!(r.limits, syn::RangeLimits::Closed(_)),
            },
            syn::Expr::Macro(_) => ExprKind::Err,
            _ => ExprKind::Err,
        };

        Expr::unhashed(kind, span)
    }

    fn lower_lit(&self, lit: &syn::Lit) -> Literal {
        match lit {
            syn::Lit::Bool(b) => Literal::Bool(b.value),
            syn::Lit::Char(c) => Literal::Char(c.value()),
            syn::Lit::Int(i) => {
                let value = i.base10_parse::<i128>().unwrap_or(0);
                Literal::Int(value, None)
            }
            syn::Lit::Float(f) => {
                let value = f.base10_parse::<f64>().unwrap_or(0.0);
                Literal::Float(value, None)
            }
            syn::Lit::Str(s) => Literal::Str(s.value()),
            syn::Lit::ByteStr(b) => Literal::ByteStr(b.value()),
            _ => Literal::Bool(false),
        }
    }

    fn lower_binop(&self, op: &syn::BinOp) -> BinaryOp {
        match op {
            syn::BinOp::Add(_) | syn::BinOp::AddAssign(_) => BinaryOp::Add,
            syn::BinOp::Sub(_) | syn::BinOp::SubAssign(_) => BinaryOp::Sub,
            syn::BinOp::Mul(_) | syn::BinOp::MulAssign(_) => BinaryOp::Mul,
            syn::BinOp::Div(_) | syn::BinOp::DivAssign(_) => BinaryOp::Div,
            syn::BinOp::Rem(_) | syn::BinOp::RemAssign(_) => BinaryOp::Rem,
            syn::BinOp::And(_) => BinaryOp::And,
            syn::BinOp::Or(_) => BinaryOp::Or,
            syn::BinOp::BitAnd(_) | syn::BinOp::BitAndAssign(_) => BinaryOp::BitAnd,
            syn::BinOp::BitOr(_) | syn::BinOp::BitOrAssign(_) => BinaryOp::BitOr,
            syn::BinOp::BitXor(_) | syn::BinOp::BitXorAssign(_) => BinaryOp::BitXor,
            syn::BinOp::Shl(_) | syn::BinOp::ShlAssign(_) => BinaryOp::Shl,
            syn::BinOp::Shr(_) | syn::BinOp::ShrAssign(_) => BinaryOp::Shr,
            syn::BinOp::Eq(_) => BinaryOp::Eq,
            syn::BinOp::Ne(_) => BinaryOp::Ne,
            syn::BinOp::Lt(_) => BinaryOp::Lt,
            syn::BinOp::Le(_) => BinaryOp::Le,
            syn::BinOp::Gt(_) => BinaryOp::Gt,
            syn::BinOp::Ge(_) => BinaryOp::Ge,
            _ => BinaryOp::Add,
        }
    }
}

use quote::ToTokens;
