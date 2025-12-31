//! Lowering from syn AST to HIR

use crate::error::Span;
use crate::hir::*;
use crate::parser::Parser;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use syn::spanned::Spanned;

pub struct Lowerer<'a> {
    parser: &'a Parser,
    file_id: u32,
    file_path: PathBuf,  // Path to current file for resolving external modules
    /// Local macro registry for expansion during lowering
    macros: std::cell::RefCell<std::collections::HashMap<String, MacroDef>>,
}

impl<'a> Lowerer<'a> {
    pub fn new(parser: &'a Parser, file_id: u32, file_path: PathBuf) -> Self {
        Self {
            parser,
            file_id,
            file_path,
            macros: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// Register a macro for expansion
    fn register_macro(&self, name: String, def: MacroDef) {
        self.macros.borrow_mut().insert(name, def);
    }

    /// Look up a macro by name
    fn lookup_macro(&self, name: &str) -> Option<MacroDef> {
        self.macros.borrow().get(name).cloned()
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

    /// Check if an item has #[cfg(test)] attribute (should be skipped during normal compilation)
    fn has_cfg_test(&self, attrs: &[syn::Attribute]) -> bool {
        for attr in attrs {
            if attr.path().is_ident("cfg") {
                let tokens = attr.meta.to_token_stream().to_string();
                if tokens.contains("test") {
                    return true;
                }
            }
        }
        false
    }

    fn span(&self, s: proc_macro2::Span) -> Span {
        // Use proc_macro2 span-locations feature to get line/column
        let start = s.start();
        let end = s.end();

        // Convert line/column to byte offset using the global source map
        if let Some(source_file) = crate::error::source_map().get_file(self.file_id) {
            let start_offset = source_file.line_col_to_offset(start.line, start.column + 1);
            let end_offset = source_file.line_col_to_offset(end.line, end.column + 1);
            Span::new(self.file_id, start_offset, end_offset)
        } else {
            // Fallback if source file not in map
            Span::new(self.file_id, 0, 0)
        }
    }

    fn alloc_def_id(&self) -> DefId {
        self.parser.next_def_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Find the file path for an external module declaration
    /// Tries: parent_dir/mod_name.rs then parent_dir/mod_name/mod.rs
    fn find_module_file(&self, mod_name: &str) -> Option<PathBuf> {
        let parent_dir = self.file_path.parent()?;

        // Try mod_name.rs first
        let file_path = parent_dir.join(format!("{}.rs", mod_name));
        if file_path.exists() {
            return Some(file_path);
        }

        // Try mod_name/mod.rs
        let dir_path = parent_dir.join(mod_name).join("mod.rs");
        if dir_path.exists() {
            return Some(dir_path);
        }

        None
    }

    /// Load an external module file and add its items to the crate
    fn load_external_module(&self, mod_path: &PathBuf, krate: &mut Crate, prefix: &str) {
        // Read and parse the file
        let content = match std::fs::read_to_string(mod_path) {
            Ok(c) => c,
            Err(_) => return,  // Silently skip if file can't be read
        };

        let file_id = self.parser.source_map.add_file(mod_path.clone(), content.clone());

        let syn_file = match syn::parse_file(&content) {
            Ok(f) => f,
            Err(_) => return,  // Silently skip parse errors
        };

        // Create a new lowerer for the module file and lower its items
        let mod_lowerer = Lowerer::new(self.parser, file_id, mod_path.clone());
        mod_lowerer.lower_items(&syn_file.items, krate, prefix);
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
                // Register macros in the macro registry for quick lookup during expansion
                if let ItemKind::Macro(ref macro_def) = hir_item.kind {
                    krate.macros.insert(hir_item.name.clone(), macro_def.clone());
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
                // Also update self_ty to use full qualified type name for module-qualified types
                // Find the impl we just inserted and fix its items
                for (_, hir_item) in krate.items.iter_mut() {
                    if let ItemKind::Impl(impl_block) = &mut hir_item.kind {
                        if let TypeKind::Path(p) = &impl_block.self_ty.kind {
                            if p.segments.first().map(|s| &s.ident) == Some(&type_name) {
                                impl_block.items = method_def_ids.clone();
                                // Update self_ty to use full qualified type name
                                if !prefix.is_empty() {
                                    impl_block.self_ty = Type {
                                        kind: TypeKind::Path(Path {
                                            segments: vec![PathSegment {
                                                ident: full_type.clone(),
                                                args: p.segments.first().and_then(|s| s.args.clone()),
                                            }],
                                        }),
                                        span: impl_block.self_ty.span,
                                    };
                                }
                                break;
                            }
                        }
                    }
                }
            }

            // Recursively lower inline modules
            if let syn::Item::Mod(m) = item {
                // Skip #[cfg(test)] modules during normal compilation
                if self.has_cfg_test(&m.attrs) {
                    continue;
                }

                if let Some((_, content_items)) = &m.content {
                    // Inline module: mod foo { ... }
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    self.lower_items(content_items, krate, &mod_prefix);
                } else {
                    // External module: mod foo; - load from foo.rs or foo/mod.rs
                    let mod_name = m.ident.to_string();
                    let mod_prefix = if prefix.is_empty() {
                        mod_name.clone()
                    } else {
                        format!("{}::{}", prefix, mod_name)
                    };

                    // Try to find the module file
                    if let Some(mod_path) = self.find_module_file(&mod_name) {
                        self.load_external_module(&mod_path, krate, &mod_prefix);
                    }
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
                syn::FnArg::Receiver(r) => {
                    // Build the base struct type
                    let struct_type = Type {
                        kind: TypeKind::Path(Path {
                            segments: vec![PathSegment {
                                ident: type_name.to_string(),
                                args: self_type_args.clone(),
                            }],
                        }),
                        span: Span::dummy(),
                    };

                    // Check if this is &self or &mut self
                    let self_type = if r.reference.is_some() {
                        let mutable = r.mutability.is_some();
                        Type {
                            kind: TypeKind::Ref {
                                lifetime: None,
                                mutable,
                                inner: Box::new(struct_type),
                            },
                            span: Span::dummy(),
                        }
                    } else {
                        struct_type
                    };

                    inputs.push(("self".to_string(), self_type));
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

        // Build struct literal with all fields copied from self
        // Point { x: self.x, y: self.y }
        let mut fields: Vec<(String, Expr)> = Vec::new();
        if let syn::Fields::Named(ref named) = s.fields {
            for field in &named.named {
                if let Some(ref ident) = field.ident {
                    let field_name = ident.to_string();
                    // Create self.field_name expression
                    let field_access = Expr::unhashed(
                        ExprKind::Field {
                            expr: Box::new(Expr::unhashed(
                                ExprKind::Path(Path {
                                    segments: vec![PathSegment { ident: "self".to_string(), args: None }],
                                }),
                                Span::new(self.file_id, 0, 0),
                            )),
                            field: field_name.clone(),
                        },
                        Span::new(self.file_id, 0, 0),
                    );
                    fields.push((field_name, field_access));
                }
            }
        }

        // Return struct literal: TypeName { fields... }
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
            syn::Item::Macro(m) => self.lower_macro(m),
            syn::Item::Use(_) => None,
            syn::Item::ExternCrate(_) => None,
            _ => None,
        }
    }

    fn lower_macro(&self, m: &syn::ItemMacro) -> Option<Item> {
        // Only handle macro_rules! definitions
        let macro_name = m.ident.as_ref()?.to_string();

        // Parse the macro body (TokenStream) into rules
        let rules = self.parse_macro_rules(&m.mac.tokens)?;
        let macro_def = MacroDef { rules };

        // Register the macro for expansion during lowering
        self.register_macro(macro_name.clone(), macro_def.clone());

        let id = self.alloc_def_id();
        Some(Item {
            id,
            name: macro_name,
            kind: ItemKind::Macro(macro_def),
            visibility: Visibility::Private, // Macros don't have visibility in syn::ItemMacro
            span: self.span(m.span()),
        })
    }

    fn parse_macro_rules(&self, tokens: &proc_macro2::TokenStream) -> Option<Vec<MacroRule>> {
        let mut rules = Vec::new();
        let mut iter = tokens.clone().into_iter().peekable();

        while iter.peek().is_some() {
            // Each rule: (pattern) => { template }
            // Skip to opening paren/bracket
            let pattern_group = match iter.next()? {
                proc_macro2::TokenTree::Group(g) => g,
                _ => continue,
            };

            // Skip =>
            match iter.next()? {
                proc_macro2::TokenTree::Punct(p) if p.as_char() == '=' => {}
                _ => continue,
            }
            match iter.next()? {
                proc_macro2::TokenTree::Punct(p) if p.as_char() == '>' => {}
                _ => continue,
            }

            // Get template group
            let template_group = match iter.next()? {
                proc_macro2::TokenTree::Group(g) => g,
                _ => continue,
            };

            // Skip optional semicolon
            if let Some(proc_macro2::TokenTree::Punct(p)) = iter.peek() {
                if p.as_char() == ';' {
                    iter.next();
                }
            }

            let pattern = self.parse_macro_tokens(pattern_group.stream());
            let template = self.parse_macro_tokens(template_group.stream());

            rules.push(MacroRule { pattern, template });
        }

        if rules.is_empty() {
            None
        } else {
            Some(rules)
        }
    }

    fn parse_macro_tokens(&self, tokens: proc_macro2::TokenStream) -> Vec<MacroToken> {
        let mut result = Vec::new();
        let mut iter = tokens.into_iter().peekable();

        while let Some(tt) = iter.next() {
            match tt {
                proc_macro2::TokenTree::Ident(ident) => {
                    result.push(MacroToken::Ident(ident.to_string()));
                }
                proc_macro2::TokenTree::Punct(p) => {
                    let ch = p.as_char();
                    if ch == '$' {
                        // Metavariable or repetition - check type first, then consume
                        let next_is_ident = matches!(iter.peek(), Some(proc_macro2::TokenTree::Ident(_)));
                        let next_is_group = matches!(iter.peek(), Some(proc_macro2::TokenTree::Group(_)));

                        if next_is_ident {
                            // Consume the ident
                            let name = if let Some(proc_macro2::TokenTree::Ident(ident)) = iter.next() {
                                ident.to_string()
                            } else {
                                unreachable!()
                            };

                            // Check for :kind
                            let has_colon = matches!(iter.peek(), Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == ':');
                            if has_colon {
                                iter.next(); // consume colon
                                if let Some(proc_macro2::TokenTree::Ident(kind_ident)) = iter.next() {
                                    let kind = match kind_ident.to_string().as_str() {
                                        "expr" => MetaVarKind::Expr,
                                        "ty" => MetaVarKind::Ty,
                                        "ident" => MetaVarKind::Ident,
                                        "pat" => MetaVarKind::Pat,
                                        "stmt" => MetaVarKind::Stmt,
                                        "block" => MetaVarKind::Block,
                                        "item" => MetaVarKind::Item,
                                        "tt" => MetaVarKind::Tt,
                                        "literal" => MetaVarKind::Literal,
                                        _ => MetaVarKind::Tt,
                                    };
                                    result.push(MacroToken::MetaVar { name, kind });
                                    continue;
                                }
                            }
                            // Just $name without :kind - treat as ident reference in template
                            result.push(MacroToken::MetaVar { name, kind: MetaVarKind::Tt });
                        } else if next_is_group {
                            // Repetition: $(...)* or $(...)+ or $(...)?
                            // Consume the group and get its stream
                            let group_stream = if let Some(proc_macro2::TokenTree::Group(g)) = iter.next() {
                                g.stream()
                            } else {
                                unreachable!()
                            };
                            let inner = self.parse_macro_tokens(group_stream);

                            // Check for separator and repetition kind
                            let mut separator = None;
                            let mut rep_kind = RepetitionKind::ZeroOrMore;

                            loop {
                                let peek_char = if let Some(proc_macro2::TokenTree::Punct(p)) = iter.peek() {
                                    Some(p.as_char())
                                } else {
                                    None
                                };

                                match peek_char {
                                    Some('*') => {
                                        rep_kind = RepetitionKind::ZeroOrMore;
                                        iter.next();
                                        break;
                                    }
                                    Some('+') => {
                                        rep_kind = RepetitionKind::OneOrMore;
                                        iter.next();
                                        break;
                                    }
                                    Some('?') => {
                                        rep_kind = RepetitionKind::ZeroOrOne;
                                        iter.next();
                                        break;
                                    }
                                    Some(c) => {
                                        // Separator
                                        separator = Some(c);
                                        iter.next();
                                    }
                                    None => break,
                                }
                            }

                            result.push(MacroToken::Repetition {
                                tokens: inner,
                                separator,
                                kind: rep_kind,
                            });
                        } else {
                            result.push(MacroToken::Punct('$'));
                        }
                    } else {
                        result.push(MacroToken::Punct(ch));
                    }
                }
                proc_macro2::TokenTree::Group(g) => {
                    let delimiter = match g.delimiter() {
                        proc_macro2::Delimiter::Parenthesis => Delimiter::Paren,
                        proc_macro2::Delimiter::Bracket => Delimiter::Bracket,
                        proc_macro2::Delimiter::Brace => Delimiter::Brace,
                        proc_macro2::Delimiter::None => Delimiter::Paren,
                    };
                    let inner = self.parse_macro_tokens(g.stream());
                    result.push(MacroToken::Group { delimiter, tokens: inner });
                }
                proc_macro2::TokenTree::Literal(lit) => {
                    result.push(MacroToken::Literal(lit.to_string()));
                }
            }
        }

        result
    }

    /// Expand a macro invocation
    fn expand_macro(&self, name: &str, tokens: &proc_macro2::TokenStream, _span: proc_macro2::Span) -> Option<syn::Expr> {
        // Look up the macro definition
        let macro_def = self.lookup_macro(name)?;

        // Convert input tokens to our MacroToken representation
        let input_tokens = self.parse_macro_tokens(tokens.clone());

        // Try each rule in order
        for rule in &macro_def.rules {
            if let Some(bindings) = self.match_macro_pattern(&rule.pattern, &input_tokens) {
                // Expand the template with bindings
                let expanded_tokens = self.expand_macro_template(&rule.template, &bindings);

                // Convert back to TokenStream and parse as expression
                let token_stream = self.macro_tokens_to_stream(&expanded_tokens);
                if let Ok(expr) = syn::parse2::<syn::Expr>(token_stream) {
                    return Some(expr);
                }
            }
        }

        None
    }

    /// Match input tokens against a macro pattern, returning bindings if matched
    fn match_macro_pattern(
        &self,
        pattern: &[MacroToken],
        input: &[MacroToken],
    ) -> Option<std::collections::HashMap<String, Vec<MacroToken>>> {
        let mut bindings: std::collections::HashMap<String, Vec<MacroToken>> = std::collections::HashMap::new();
        let mut pattern_idx = 0;
        let mut input_idx = 0;

        while pattern_idx < pattern.len() {
            match &pattern[pattern_idx] {
                MacroToken::MetaVar { name, kind: _ } => {
                    // Capture the next input token(s) into this metavariable
                    if input_idx >= input.len() {
                        return None; // No more input
                    }
                    // For simplicity, capture a single token (or group) for now
                    bindings.insert(name.clone(), vec![input[input_idx].clone()]);
                    pattern_idx += 1;
                    input_idx += 1;
                }
                MacroToken::Repetition { tokens: rep_pattern, separator, kind } => {
                    // Handle repetition patterns
                    let mut rep_bindings: std::collections::HashMap<String, Vec<MacroToken>> = std::collections::HashMap::new();

                    // Collect metavar names in the repetition pattern
                    for tok in rep_pattern {
                        if let MacroToken::MetaVar { name, .. } = tok {
                            rep_bindings.insert(name.clone(), Vec::new());
                        }
                    }

                    // Match repeated elements
                    let mut first = true;
                    while input_idx < input.len() {
                        // Check for separator
                        if !first {
                            if let Some(sep) = separator {
                                if let Some(MacroToken::Punct(c)) = input.get(input_idx) {
                                    if c == sep {
                                        input_idx += 1;
                                    } else {
                                        break; // No separator, end of repetition
                                    }
                                } else {
                                    break;
                                }
                            }
                        }
                        first = false;

                        // Try to match the repetition pattern
                        if input_idx >= input.len() {
                            break;
                        }

                        // Simple case: single metavar in repetition
                        if rep_pattern.len() == 1 {
                            if let MacroToken::MetaVar { name, .. } = &rep_pattern[0] {
                                rep_bindings.get_mut(name).unwrap().push(input[input_idx].clone());
                                input_idx += 1;
                                continue;
                            }
                        }

                        // Try to match the pattern
                        let remaining_input = &input[input_idx..];
                        if let Some(sub_bindings) = self.match_macro_pattern(rep_pattern, &[remaining_input[0].clone()]) {
                            for (k, v) in sub_bindings {
                                rep_bindings.get_mut(&k).map(|vec| vec.extend(v));
                            }
                            input_idx += 1;
                        } else {
                            break; // Pattern didn't match
                        }
                    }

                    // Check kind constraints
                    let count = rep_bindings.values().next().map(|v| v.len()).unwrap_or(0);
                    match kind {
                        RepetitionKind::OneOrMore => {
                            if count == 0 { return None; }
                        }
                        RepetitionKind::ZeroOrOne => {
                            if count > 1 { return None; }
                        }
                        _ => {}
                    }

                    // Merge repetition bindings
                    for (k, v) in rep_bindings {
                        bindings.insert(k, v);
                    }
                    pattern_idx += 1;
                }
                MacroToken::Ident(expected) => {
                    if let Some(MacroToken::Ident(actual)) = input.get(input_idx) {
                        if expected != actual {
                            return None;
                        }
                    } else {
                        return None;
                    }
                    pattern_idx += 1;
                    input_idx += 1;
                }
                MacroToken::Punct(expected) => {
                    if let Some(MacroToken::Punct(actual)) = input.get(input_idx) {
                        if expected != actual {
                            return None;
                        }
                    } else {
                        return None;
                    }
                    pattern_idx += 1;
                    input_idx += 1;
                }
                MacroToken::Literal(expected) => {
                    if let Some(MacroToken::Literal(actual)) = input.get(input_idx) {
                        if expected != actual {
                            return None;
                        }
                    } else {
                        return None;
                    }
                    pattern_idx += 1;
                    input_idx += 1;
                }
                MacroToken::Group { delimiter, tokens: group_pattern } => {
                    if let Some(MacroToken::Group { delimiter: input_delim, tokens: input_group }) = input.get(input_idx) {
                        if delimiter != input_delim {
                            return None;
                        }
                        // Recursively match the group contents
                        if let Some(sub_bindings) = self.match_macro_pattern(group_pattern, input_group) {
                            for (k, v) in sub_bindings {
                                bindings.insert(k, v);
                            }
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                    pattern_idx += 1;
                    input_idx += 1;
                }
            }
        }

        // All input should be consumed
        if input_idx == input.len() {
            Some(bindings)
        } else {
            None
        }
    }

    /// Expand a macro template with bindings
    fn expand_macro_template(
        &self,
        template: &[MacroToken],
        bindings: &std::collections::HashMap<String, Vec<MacroToken>>,
    ) -> Vec<MacroToken> {
        let mut result = Vec::new();

        for token in template {
            match token {
                MacroToken::MetaVar { name, .. } => {
                    // Substitute the bound value
                    if let Some(bound) = bindings.get(name) {
                        if bound.len() == 1 {
                            result.push(bound[0].clone());
                        } else {
                            result.extend(bound.iter().cloned());
                        }
                    }
                }
                MacroToken::Repetition { tokens: rep_template, separator, .. } => {
                    // Find the metavar names in the repetition and get their count
                    let mut metavar_names = Vec::new();
                    Self::collect_metavar_names(rep_template, &mut metavar_names);

                    let count = metavar_names.first()
                        .and_then(|n| bindings.get(n))
                        .map(|v| v.len())
                        .unwrap_or(0);

                    for i in 0..count {
                        if i > 0 {
                            if let Some(sep) = separator {
                                result.push(MacroToken::Punct(*sep));
                            }
                        }
                        // Expand the repetition template with the i-th element of each binding
                        let mut index_bindings = std::collections::HashMap::new();
                        for name in &metavar_names {
                            if let Some(values) = bindings.get(name) {
                                if i < values.len() {
                                    index_bindings.insert(name.clone(), vec![values[i].clone()]);
                                }
                            }
                        }
                        let expanded = self.expand_macro_template(rep_template, &index_bindings);
                        result.extend(expanded);
                    }
                }
                MacroToken::Group { delimiter, tokens } => {
                    let expanded = self.expand_macro_template(tokens, bindings);
                    result.push(MacroToken::Group {
                        delimiter: *delimiter,
                        tokens: expanded,
                    });
                }
                other => {
                    result.push(other.clone());
                }
            }
        }

        result
    }

    /// Collect metavar names from a token sequence
    fn collect_metavar_names(tokens: &[MacroToken], names: &mut Vec<String>) {
        for token in tokens {
            match token {
                MacroToken::MetaVar { name, .. } => {
                    if !names.contains(name) {
                        names.push(name.clone());
                    }
                }
                MacroToken::Group { tokens, .. } => {
                    Self::collect_metavar_names(tokens, names);
                }
                MacroToken::Repetition { tokens, .. } => {
                    Self::collect_metavar_names(tokens, names);
                }
                _ => {}
            }
        }
    }

    /// Convert MacroTokens back to a TokenStream
    fn macro_tokens_to_stream(&self, tokens: &[MacroToken]) -> proc_macro2::TokenStream {
        let mut stream = proc_macro2::TokenStream::new();

        for token in tokens {
            match token {
                MacroToken::Ident(s) => {
                    let ident = proc_macro2::Ident::new(s, proc_macro2::Span::call_site());
                    stream.extend(quote::quote!(#ident));
                }
                MacroToken::Punct(c) => {
                    let punct = proc_macro2::Punct::new(*c, proc_macro2::Spacing::Alone);
                    stream.extend(std::iter::once(proc_macro2::TokenTree::Punct(punct)));
                }
                MacroToken::Literal(s) => {
                    // Parse the literal string back to a token
                    if let Ok(lit) = syn::parse_str::<proc_macro2::Literal>(s) {
                        stream.extend(std::iter::once(proc_macro2::TokenTree::Literal(lit)));
                    } else {
                        // Try as ident if not a valid literal
                        let ident = proc_macro2::Ident::new(s, proc_macro2::Span::call_site());
                        stream.extend(quote::quote!(#ident));
                    }
                }
                MacroToken::MetaVar { name, .. } => {
                    // This shouldn't happen after expansion, but handle it
                    let ident = proc_macro2::Ident::new(name, proc_macro2::Span::call_site());
                    stream.extend(quote::quote!(#ident));
                }
                MacroToken::Group { delimiter, tokens } => {
                    let inner = self.macro_tokens_to_stream(tokens);
                    let delim = match delimiter {
                        Delimiter::Paren => proc_macro2::Delimiter::Parenthesis,
                        Delimiter::Bracket => proc_macro2::Delimiter::Bracket,
                        Delimiter::Brace => proc_macro2::Delimiter::Brace,
                    };
                    let group = proc_macro2::Group::new(delim, inner);
                    stream.extend(std::iter::once(proc_macro2::TokenTree::Group(group)));
                }
                MacroToken::Repetition { tokens, separator, .. } => {
                    // Already expanded, just emit the tokens
                    for (i, tok) in tokens.iter().enumerate() {
                        if i > 0 {
                            if let Some(sep) = separator {
                                let punct = proc_macro2::Punct::new(*sep, proc_macro2::Spacing::Alone);
                                stream.extend(std::iter::once(proc_macro2::TokenTree::Punct(punct)));
                            }
                        }
                        stream.extend(self.macro_tokens_to_stream(&[tok.clone()]));
                    }
                }
            }
        }

        stream
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
                syn::ImplItem::Fn(_f) => Some(self.alloc_def_id()),
                _ => None,
            })
            .collect();

        // Extract associated type bindings from impl items
        let mut assoc_types = Vec::new();
        for item in &i.items {
            if let syn::ImplItem::Type(type_item) = item {
                assoc_types.push(AssocTypeBinding {
                    name: type_item.ident.to_string(),
                    ty: self.lower_type(&type_item.ty),
                    span: self.span(type_item.span()),
                });
            }
        }

        Item {
            id,
            name: String::new(),
            kind: ItemKind::Impl(Impl {
                generics: self.lower_generics(&i.generics),
                trait_ref,
                self_ty,
                items,
                assoc_types,
            }),
            visibility: Visibility::Private,
            span: self.span(i.span()),
        }
    }

    fn lower_trait(&self, t: &syn::ItemTrait) -> Item {
        let id = self.alloc_def_id();

        // Extract associated types from trait items
        let mut assoc_types = Vec::new();
        for item in &t.items {
            if let syn::TraitItem::Type(type_item) = item {
                assoc_types.push(AssocTypeDecl {
                    name: type_item.ident.to_string(),
                    bounds: type_item.bounds.iter()
                        .filter_map(|b| self.lower_type_param_bound(b))
                        .collect(),
                    default: type_item.default.as_ref().map(|(_, ty)| self.lower_type(ty)),
                    span: self.span(type_item.span()),
                });
            }
        }

        Item {
            id,
            name: t.ident.to_string(),
            kind: ItemKind::Trait(Trait {
                generics: self.lower_generics(&t.generics),
                bounds: t.supertraits.iter().filter_map(|b| self.lower_type_param_bound(b)).collect(),
                items: Vec::new(),
                assoc_types,
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
            syn::Type::ImplTrait(it) => {
                let bounds = it.bounds.iter()
                    .filter_map(|b| self.lower_type_param_bound(b))
                    .collect();
                TypeKind::ImplTrait(bounds)
            }
            syn::Type::TraitObject(to) => {
                let bounds = to.bounds.iter()
                    .filter_map(|b| self.lower_type_param_bound(b))
                    .collect();
                TypeKind::DynTrait(bounds)
            }
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
            syn::Pat::Type(pt) => {
                // Handle typed patterns like `x: i32` in closures
                // Extract the inner pattern (the variable name)
                return self.lower_pattern(&pt.pat);
            }
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
            syn::Expr::Macro(m) => {
                // Extract macro name
                let macro_name = m.mac.path.segments.last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();

                // Try to expand the macro
                if let Some(expanded) = self.expand_macro(&macro_name, &m.mac.tokens, expr.span()) {
                    return self.lower_expr(&expanded);
                }
                // Fallback: couldn't expand
                ExprKind::Err
            }
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
