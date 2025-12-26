//! Lower parser2 AST to HIR
//!
//! This module converts the Chumsky-parsed AST into the HIR used by
//! type checking and code generation.

use crate::error::Span as HirSpan;
use crate::hir::{
    self, AssocTypeBinding, AssocTypeDecl, Arm, BinaryOp, Block, Const, Crate, DefId,
    Enum, Expr, ExprKind, Field, FloatType, FnSig, Function, GenericArg, GenericArgs,
    GenericParam, Generics, Impl, IntType, Item, ItemKind, Literal, MacroDef, MacroRule,
    MacroToken, MetaVarKind, Module, Path, PathSegment, Pattern, PatternKind, RepetitionKind,
    Static, Stmt, StmtKind, Struct, StructKind, Trait, Type, TypeAlias, TypeBound, TypeKind,
    UintType, UnaryOp, Variant, Visibility, WherePredicate, FieldPattern, Delimiter,
};
use crate::lexer::Span as LexerSpan;
use crate::parser2::ast;
use indexmap::IndexMap;
use std::collections::HashMap;

/// Context for lowering AST to HIR
pub struct LowerContext {
    /// The HIR crate being built
    pub hir_crate: Crate,
    /// Map from names to DefIds
    def_ids: HashMap<String, DefId>,
    /// Next DefId to allocate
    next_def_id: u32,
    /// Current module path prefix
    module_prefix: String,
}

impl LowerContext {
    pub fn new(crate_name: String) -> Self {
        Self {
            hir_crate: Crate::new(crate_name),
            def_ids: HashMap::new(),
            next_def_id: 0,
            module_prefix: String::new(),
        }
    }

    /// Allocate a new DefId
    fn alloc_def_id(&mut self) -> DefId {
        let id = self.next_def_id;
        self.next_def_id += 1;
        id
    }

    /// Convert lexer span to HIR span
    fn convert_span(&self, span: LexerSpan) -> HirSpan {
        HirSpan::new(0, span.start, span.end)
    }

    /// Lower a complete source file to HIR
    pub fn lower_source_file(&mut self, source: &ast::SourceFile) -> &Crate {
        for item in &source.items {
            self.lower_item(item);
        }
        &self.hir_crate
    }

    /// Lower a single item
    fn lower_item(&mut self, item: &ast::Item) -> Option<DefId> {
        match &item.kind {
            ast::ItemKind::Function(f) => Some(self.lower_function(f, item.span)),
            ast::ItemKind::Struct(s) => Some(self.lower_struct(s, item.span)),
            ast::ItemKind::Enum(e) => Some(self.lower_enum(e, item.span)),
            ast::ItemKind::Impl(i) => Some(self.lower_impl(i, item.span)),
            ast::ItemKind::Trait(t) => Some(self.lower_trait(t, item.span)),
            ast::ItemKind::TypeAlias(ta) => Some(self.lower_type_alias(ta, item.span)),
            ast::ItemKind::Const(c) => Some(self.lower_const(c, item.span)),
            ast::ItemKind::Static(s) => Some(self.lower_static(s, item.span)),
            ast::ItemKind::Module(m) => Some(self.lower_module(m, item.span)),
            ast::ItemKind::Use(u) => {
                self.lower_use(u);
                None
            }
            ast::ItemKind::MacroRules(m) => Some(self.lower_macro_rules(m, item.span)),
        }
    }

    fn lower_function(&mut self, f: &ast::Function, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let name = self.qualified_name(&f.name.name);

        // Check for main entry point
        if f.name.name == "main" && self.module_prefix.is_empty() {
            self.hir_crate.entry_point = Some(def_id);
        }

        let inputs: Vec<(String, Type)> = f.params.iter().map(|p| {
            let name = self.pattern_name(&p.pattern);
            let ty = self.lower_type(&p.ty);
            (name, ty)
        }).collect();

        let output = f.ret_type.as_ref()
            .map(|t| self.lower_type(t))
            .unwrap_or_else(|| Type {
                kind: TypeKind::Unit,
                span: self.convert_span(span),
            });

        let generics = self.lower_generics(&f.generics);

        let body = f.body.as_ref().map(|b| self.lower_block(b));

        let func = Function {
            sig: FnSig { inputs, output, generics },
            body,
            is_async: f.is_async,
            is_const: false,
            is_unsafe: f.is_unsafe,
        };

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::Function(func),
            visibility: if f.is_pub { Visibility::Public } else { Visibility::Private },
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_struct(&mut self, s: &ast::Struct, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let name = self.qualified_name(&s.name.name);
        let generics = self.lower_generics(&s.generics);

        let kind = match &s.fields {
            ast::StructFields::Named(fields) => {
                let hir_fields: Vec<Field> = fields.iter().map(|f| Field {
                    name: f.name.name.clone(),
                    ty: self.lower_type(&f.ty),
                    visibility: if f.is_pub { Visibility::Public } else { Visibility::Private },
                }).collect();
                StructKind::Named(hir_fields)
            }
            ast::StructFields::Tuple(types) => {
                let hir_types: Vec<Type> = types.iter().map(|t| self.lower_type(t)).collect();
                StructKind::Tuple(hir_types)
            }
            ast::StructFields::Unit => StructKind::Unit,
        };

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::Struct(Struct { generics, kind }),
            visibility: if s.is_pub { Visibility::Public } else { Visibility::Private },
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_enum(&mut self, e: &ast::Enum, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let name = self.qualified_name(&e.name.name);
        let generics = self.lower_generics(&e.generics);

        let variants: Vec<Variant> = e.variants.iter().map(|v| {
            let kind = match &v.fields {
                ast::StructFields::Named(fields) => {
                    let hir_fields: Vec<Field> = fields.iter().map(|f| Field {
                        name: f.name.name.clone(),
                        ty: self.lower_type(&f.ty),
                        visibility: Visibility::Public,
                    }).collect();
                    StructKind::Named(hir_fields)
                }
                ast::StructFields::Tuple(types) => {
                    let hir_types: Vec<Type> = types.iter().map(|t| self.lower_type(t)).collect();
                    StructKind::Tuple(hir_types)
                }
                ast::StructFields::Unit => StructKind::Unit,
            };
            Variant {
                name: v.name.name.clone(),
                kind,
                discriminant: v.discriminant.as_ref().map(|e| self.lower_expr(e)),
            }
        }).collect();

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::Enum(Enum { generics, variants }),
            visibility: if e.is_pub { Visibility::Public } else { Visibility::Private },
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_impl(&mut self, i: &ast::Impl, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let generics = self.lower_generics(&i.generics);
        let trait_ref = i.trait_.as_ref().map(|p| self.lower_path(p));
        let self_ty = self.lower_type(&i.self_ty);

        // Get type name for naming
        let type_name = self.type_name(&i.self_ty);
        let trait_name = i.trait_.as_ref().map(|p| self.path_to_string(p));

        let name = match &trait_name {
            Some(t) => format!("<{} as {}>", type_name, t),
            None => format!("<{}>", type_name),
        };

        // Lower impl items
        let mut item_ids = Vec::new();
        let mut assoc_types = Vec::new();

        for impl_item in &i.items {
            match &impl_item.kind {
                ast::ImplItemKind::Function(f) => {
                    // Name method as Type_method or Type::Trait_method
                    let method_name = match &trait_name {
                        Some(_t) => format!("{}_{}", type_name, f.name.name),
                        None => format!("{}_{}", type_name, f.name.name),
                    };

                    let method_def_id = self.alloc_def_id();

                    // Resolve Self to the actual impl type
                    let resolve_self = |ty: Type, self_ty: &Type| -> Type {
                        match &ty.kind {
                            TypeKind::Path(path) if path.segments.len() == 1
                                && path.segments[0].ident == "Self" => {
                                self_ty.clone()
                            }
                            TypeKind::Ref { lifetime, mutable, inner } => {
                                if let TypeKind::Path(path) = &inner.kind {
                                    if path.segments.len() == 1 && path.segments[0].ident == "Self" {
                                        return Type {
                                            kind: TypeKind::Ref {
                                                lifetime: lifetime.clone(),
                                                mutable: *mutable,
                                                inner: Box::new(self_ty.clone()),
                                            },
                                            span: ty.span,
                                        };
                                    }
                                }
                                ty
                            }
                            _ => ty,
                        }
                    };

                    let inputs: Vec<(String, Type)> = f.params.iter().map(|p| {
                        let pname = self.pattern_name(&p.pattern);
                        let ty = self.lower_type(&p.ty);
                        let ty = resolve_self(ty, &self_ty);
                        (pname, ty)
                    }).collect();

                    let output = f.ret_type.as_ref()
                        .map(|t| {
                            let ty = self.lower_type(t);
                            resolve_self(ty, &self_ty)
                        })
                        .unwrap_or_else(|| Type {
                            kind: TypeKind::Unit,
                            span: self.convert_span(impl_item.span),
                        });

                    let method_generics = self.lower_generics(&f.generics);
                    let body = f.body.as_ref().map(|b| self.lower_block(b));

                    let func = Function {
                        sig: FnSig { inputs, output, generics: method_generics },
                        body,
                        is_async: f.is_async,
                        is_const: false,
                        is_unsafe: f.is_unsafe,
                    };

                    let method_item = Item {
                        id: method_def_id,
                        name: method_name.clone(),
                        kind: ItemKind::Function(func),
                        visibility: Visibility::Public,
                        span: self.convert_span(impl_item.span),
                    };

                    self.def_ids.insert(method_name, method_def_id);
                    self.hir_crate.items.insert(method_def_id, method_item);
                    item_ids.push(method_def_id);
                }
                ast::ImplItemKind::TypeAlias(ta) => {
                    if let Some(ty) = &ta.ty {
                        assoc_types.push(AssocTypeBinding {
                            name: ta.name.name.clone(),
                            ty: self.lower_type(ty),
                            span: self.convert_span(impl_item.span),
                        });
                    }
                }
                ast::ImplItemKind::Const(_c) => {
                    // TODO: Handle const in impl
                }
            }
        }

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::Impl(Impl {
                generics,
                trait_ref,
                self_ty,
                items: item_ids,
                assoc_types,
            }),
            visibility: Visibility::Public,
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_trait(&mut self, t: &ast::Trait, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let name = self.qualified_name(&t.name.name);
        let generics = self.lower_generics(&t.generics);

        let bounds: Vec<TypeBound> = t.bounds.iter().map(|b| self.lower_type_bound(b)).collect();

        let mut item_ids = Vec::new();
        let mut assoc_types = Vec::new();

        for trait_item in &t.items {
            match &trait_item.kind {
                ast::TraitItemKind::Function(f) => {
                    let method_name = format!("{}::{}", name, f.name.name);
                    let method_def_id = self.alloc_def_id();

                    let inputs: Vec<(String, Type)> = f.params.iter().map(|p| {
                        let pname = self.pattern_name(&p.pattern);
                        let ty = self.lower_type(&p.ty);
                        (pname, ty)
                    }).collect();

                    let output = f.ret_type.as_ref()
                        .map(|ty| self.lower_type(ty))
                        .unwrap_or_else(|| Type {
                            kind: TypeKind::Unit,
                            span: self.convert_span(trait_item.span),
                        });

                    let method_generics = self.lower_generics(&f.generics);
                    let body = f.body.as_ref().map(|b| self.lower_block(b));

                    let func = Function {
                        sig: FnSig { inputs, output, generics: method_generics },
                        body,
                        is_async: f.is_async,
                        is_const: false,
                        is_unsafe: f.is_unsafe,
                    };

                    let method_item = Item {
                        id: method_def_id,
                        name: method_name.clone(),
                        kind: ItemKind::Function(func),
                        visibility: Visibility::Public,
                        span: self.convert_span(trait_item.span),
                    };

                    self.def_ids.insert(method_name, method_def_id);
                    self.hir_crate.items.insert(method_def_id, method_item);
                    item_ids.push(method_def_id);
                }
                ast::TraitItemKind::TypeAlias(ta) => {
                    assoc_types.push(AssocTypeDecl {
                        name: ta.name.name.clone(),
                        bounds: vec![],
                        default: ta.ty.as_ref().map(|ty| self.lower_type(ty)),
                        span: self.convert_span(trait_item.span),
                    });
                }
                ast::TraitItemKind::Const(_c) => {
                    // TODO: Handle const in trait
                }
            }
        }

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::Trait(Trait {
                generics,
                bounds,
                items: item_ids,
                assoc_types,
            }),
            visibility: if t.is_pub { Visibility::Public } else { Visibility::Private },
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_type_alias(&mut self, ta: &ast::TypeAlias, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let name = self.qualified_name(&ta.name.name);
        let generics = self.lower_generics(&ta.generics);

        let ty = ta.ty.as_ref()
            .map(|t| self.lower_type(t))
            .unwrap_or_else(|| Type {
                kind: TypeKind::Error,
                span: self.convert_span(span),
            });

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::TypeAlias(TypeAlias { generics, ty }),
            visibility: if ta.is_pub { Visibility::Public } else { Visibility::Private },
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_const(&mut self, c: &ast::Const, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let name = self.qualified_name(&c.name.name);

        let ty = self.lower_type(&c.ty);
        let value = c.value.as_ref()
            .map(|e| self.lower_expr(e))
            .unwrap_or_else(|| Expr::new(ExprKind::Err, self.convert_span(span)));

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::Const(Const { ty, value }),
            visibility: if c.is_pub { Visibility::Public } else { Visibility::Private },
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_static(&mut self, s: &ast::Static, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let name = self.qualified_name(&s.name.name);

        let ty = self.lower_type(&s.ty);
        let value = s.value.as_ref()
            .map(|e| self.lower_expr(e))
            .unwrap_or_else(|| Expr::new(ExprKind::Err, self.convert_span(span)));

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::Static(Static { ty, value, mutable: s.is_mut }),
            visibility: if s.is_pub { Visibility::Public } else { Visibility::Private },
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_module(&mut self, m: &ast::Module, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let name = self.qualified_name(&m.name.name);

        let old_prefix = self.module_prefix.clone();
        self.module_prefix = if old_prefix.is_empty() {
            m.name.name.clone()
        } else {
            format!("{}::{}", old_prefix, m.name.name)
        };

        let mut item_ids = Vec::new();
        if let Some(items) = &m.items {
            for item in items {
                if let Some(id) = self.lower_item(item) {
                    item_ids.push(id);
                }
            }
        }

        self.module_prefix = old_prefix;

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::Module(Module { items: item_ids }),
            visibility: if m.is_pub { Visibility::Public } else { Visibility::Private },
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_use(&mut self, u: &ast::Use) {
        self.lower_use_tree(&u.tree, String::new());
    }

    fn lower_use_tree(&mut self, tree: &ast::UseTree, prefix: String) {
        match tree {
            ast::UseTree::Path(path, subtree) => {
                let path_str = self.path_to_string(path);
                let full_path = if prefix.is_empty() {
                    path_str.clone()
                } else {
                    format!("{}::{}", prefix, path_str)
                };

                if let Some(sub) = subtree {
                    self.lower_use_tree(sub, full_path);
                } else {
                    // Simple use: use path::to::item;
                    if let Some(last) = path.segments.last() {
                        self.hir_crate.imports.insert(last.ident.name.clone(), full_path);
                    }
                }
            }
            ast::UseTree::Glob => {
                // use path::*; - we don't fully support this yet
            }
            ast::UseTree::Group(trees) => {
                for tree in trees {
                    self.lower_use_tree(tree, prefix.clone());
                }
            }
            ast::UseTree::Alias(orig, alias) => {
                let full_path = if prefix.is_empty() {
                    orig.name.clone()
                } else {
                    format!("{}::{}", prefix, orig.name)
                };
                self.hir_crate.imports.insert(alias.name.clone(), full_path);
            }
        }
    }

    fn lower_macro_rules(&mut self, m: &ast::MacroRules, span: LexerSpan) -> DefId {
        let def_id = self.alloc_def_id();
        let name = self.qualified_name(&m.name.name);

        let rules: Vec<MacroRule> = m.rules.iter().map(|r| {
            MacroRule {
                pattern: r.pattern.iter().map(|t| self.lower_macro_token(t)).collect(),
                template: r.body.iter().map(|t| self.lower_macro_token(t)).collect(),
            }
        }).collect();

        let macro_def = MacroDef { rules };
        self.hir_crate.macros.insert(m.name.name.clone(), macro_def.clone());

        let item = Item {
            id: def_id,
            name: name.clone(),
            kind: ItemKind::Macro(macro_def),
            visibility: Visibility::Public,
            span: self.convert_span(span),
        };

        self.def_ids.insert(name, def_id);
        self.hir_crate.items.insert(def_id, item);
        def_id
    }

    fn lower_macro_token(&self, token: &ast::MacroToken) -> MacroToken {
        match token {
            ast::MacroToken::Token(s) => MacroToken::Ident(s.clone()),
            ast::MacroToken::Var(name, kind) => {
                let mk = kind.as_ref().map(|k| match k.as_str() {
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
                }).unwrap_or(MetaVarKind::Tt);
                MacroToken::MetaVar { name: name.clone(), kind: mk }
            }
            ast::MacroToken::Repeat(tokens, sep, kind) => {
                let rk = match kind {
                    ast::RepeatKind::ZeroOrMore => RepetitionKind::ZeroOrMore,
                    ast::RepeatKind::OneOrMore => RepetitionKind::OneOrMore,
                    ast::RepeatKind::ZeroOrOne => RepetitionKind::ZeroOrOne,
                };
                MacroToken::Repetition {
                    tokens: tokens.iter().map(|t| self.lower_macro_token(t)).collect(),
                    separator: *sep,
                    kind: rk,
                }
            }
        }
    }

    // ========================================================================
    // Type lowering
    // ========================================================================

    fn lower_type(&self, ty: &ast::Type) -> Type {
        let kind = match &ty.kind {
            ast::TypeKind::Path(path) => self.lower_type_path(path),
            ast::TypeKind::Ref { lifetime, mutable, inner } => TypeKind::Ref {
                lifetime: lifetime.as_ref().map(|l| l.name.clone()),
                mutable: *mutable,
                inner: Box::new(self.lower_type(inner)),
            },
            ast::TypeKind::Ptr { mutable, inner } => TypeKind::Ptr {
                mutable: *mutable,
                inner: Box::new(self.lower_type(inner)),
            },
            ast::TypeKind::Slice(inner) => TypeKind::Slice(Box::new(self.lower_type(inner))),
            ast::TypeKind::Array(elem, _len) => {
                // TODO: Evaluate const expr for length
                TypeKind::Array {
                    elem: Box::new(self.lower_type(elem)),
                    len: 0, // Placeholder
                }
            }
            ast::TypeKind::Tuple(types) => {
                if types.is_empty() {
                    TypeKind::Unit
                } else {
                    TypeKind::Tuple(types.iter().map(|t| self.lower_type(t)).collect())
                }
            }
            ast::TypeKind::Fn { params, ret } => TypeKind::Fn {
                inputs: params.iter().map(|t| self.lower_type(t)).collect(),
                output: Box::new(self.lower_type(ret)),
            },
            ast::TypeKind::Never => TypeKind::Never,
            ast::TypeKind::Infer => TypeKind::Infer,
            ast::TypeKind::ImplTrait(bounds) => {
                TypeKind::ImplTrait(bounds.iter().map(|b| self.lower_type_bound(b)).collect())
            }
            ast::TypeKind::DynTrait(bounds) => {
                TypeKind::DynTrait(bounds.iter().map(|b| self.lower_type_bound(b)).collect())
            }
        };
        Type { kind, span: self.convert_span(ty.span) }
    }

    fn lower_type_path(&self, path: &ast::Path) -> TypeKind {
        // Handle primitive types
        if path.segments.len() == 1 {
            let name = &path.segments[0].ident.name;
            match name.as_str() {
                "bool" => return TypeKind::Bool,
                "char" => return TypeKind::Char,
                "str" => return TypeKind::Str,
                "i8" => return TypeKind::Int(IntType::I8),
                "i16" => return TypeKind::Int(IntType::I16),
                "i32" => return TypeKind::Int(IntType::I32),
                "i64" => return TypeKind::Int(IntType::I64),
                "i128" => return TypeKind::Int(IntType::I128),
                "isize" => return TypeKind::Int(IntType::Isize),
                "u8" => return TypeKind::Uint(UintType::U8),
                "u16" => return TypeKind::Uint(UintType::U16),
                "u32" => return TypeKind::Uint(UintType::U32),
                "u64" => return TypeKind::Uint(UintType::U64),
                "u128" => return TypeKind::Uint(UintType::U128),
                "usize" => return TypeKind::Uint(UintType::Usize),
                "f32" => return TypeKind::Float(FloatType::F32),
                "f64" => return TypeKind::Float(FloatType::F64),
                _ => {}
            }
        }
        TypeKind::Path(self.lower_path(path))
    }

    fn lower_type_bound(&self, bound: &ast::TypeBound) -> TypeBound {
        match bound {
            ast::TypeBound::Trait(path) => TypeBound::Trait(self.lower_path(path)),
            ast::TypeBound::Lifetime(lt) => TypeBound::Lifetime(lt.name.clone()),
        }
    }

    // ========================================================================
    // Generics lowering
    // ========================================================================

    fn lower_generics(&self, generics: &ast::Generics) -> Generics {
        let params: Vec<GenericParam> = generics.params.iter().map(|p| {
            match p {
                ast::GenericParam::Type { name, bounds, default: _ } => {
                    GenericParam::Type {
                        name: name.name.clone(),
                        bounds: bounds.iter().map(|b| self.lower_type_bound(b)).collect(),
                    }
                }
                ast::GenericParam::Lifetime { name, bounds: _ } => {
                    GenericParam::Lifetime { name: name.name.clone() }
                }
                ast::GenericParam::Const { name, ty, default: _ } => {
                    GenericParam::Const {
                        name: name.name.clone(),
                        ty: self.lower_type(ty),
                    }
                }
            }
        }).collect();

        let where_clause: Vec<WherePredicate> = generics.where_clause
            .as_ref()
            .map(|wc| {
                wc.predicates.iter().filter_map(|p| {
                    match p {
                        ast::WherePredicate::Type { ty, bounds } => Some(WherePredicate {
                            ty: self.lower_type(ty),
                            bounds: bounds.iter().map(|b| self.lower_type_bound(b)).collect(),
                        }),
                        ast::WherePredicate::Lifetime { .. } => None,
                    }
                }).collect()
            })
            .unwrap_or_default();

        Generics { params, where_clause }
    }

    // ========================================================================
    // Path lowering
    // ========================================================================

    fn lower_path(&self, path: &ast::Path) -> Path {
        Path {
            segments: path.segments.iter().map(|seg| {
                PathSegment {
                    ident: seg.ident.name.clone(),
                    args: seg.args.as_ref().map(|a| self.lower_generic_args(a)),
                }
            }).collect(),
        }
    }

    fn lower_generic_args(&self, args: &ast::GenericArgs) -> GenericArgs {
        GenericArgs {
            args: args.args.iter().map(|a| {
                match a {
                    ast::GenericArg::Type(ty) => GenericArg::Type(self.lower_type(ty)),
                    ast::GenericArg::Lifetime(lt) => GenericArg::Lifetime(lt.name.clone()),
                    ast::GenericArg::Const(expr) => GenericArg::Const(self.lower_expr(expr)),
                }
            }).collect(),
        }
    }

    // ========================================================================
    // Block and statement lowering
    // ========================================================================

    fn lower_block(&self, block: &ast::Block) -> Block {
        let mut stmts = Vec::new();
        let mut expr = None;

        for (i, stmt) in block.stmts.iter().enumerate() {
            let is_last = i == block.stmts.len() - 1;

            match &stmt.kind {
                ast::StmtKind::Let { pat, ty, init } => {
                    stmts.push(Stmt {
                        kind: StmtKind::Let {
                            pattern: self.lower_pattern(pat),
                            ty: ty.as_ref().map(|t| self.lower_type(t)),
                            init: init.as_ref().map(|e| self.lower_expr(e)),
                        },
                        span: self.convert_span(stmt.span),
                    });
                }
                ast::StmtKind::Expr(e) => {
                    stmts.push(Stmt {
                        kind: StmtKind::Semi(self.lower_expr(e)),
                        span: self.convert_span(stmt.span),
                    });
                }
                ast::StmtKind::ExprNoSemi(e) => {
                    if is_last {
                        expr = Some(Box::new(self.lower_expr(e)));
                    } else {
                        stmts.push(Stmt {
                            kind: StmtKind::Expr(self.lower_expr(e)),
                            span: self.convert_span(stmt.span),
                        });
                    }
                }
                ast::StmtKind::Item(_) => {
                    // Items in blocks - skip for now
                }
                ast::StmtKind::Empty => {}
            }
        }

        Block {
            stmts,
            expr,
            span: self.convert_span(block.span),
        }
    }

    // ========================================================================
    // Expression lowering
    // ========================================================================

    fn lower_expr(&self, expr: &ast::Expr) -> Expr {
        let span = self.convert_span(expr.span);
        let kind = match &expr.kind {
            ast::ExprKind::Lit(lit) => ExprKind::Lit(self.lower_literal(lit)),
            ast::ExprKind::Path(path) => ExprKind::Path(self.lower_path(path)),
            ast::ExprKind::Binary { op, left, right } => ExprKind::Binary {
                op: self.lower_binop(*op),
                lhs: Box::new(self.lower_expr(left)),
                rhs: Box::new(self.lower_expr(right)),
            },
            ast::ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
                op: self.lower_unaryop(*op),
                expr: Box::new(self.lower_expr(inner)),
            },
            ast::ExprKind::Call { func, args } => ExprKind::Call {
                func: Box::new(self.lower_expr(func)),
                args: args.iter().map(|a| self.lower_expr(a)).collect(),
            },
            ast::ExprKind::MethodCall { receiver, method, turbofish: _, args } => ExprKind::MethodCall {
                receiver: Box::new(self.lower_expr(receiver)),
                method: method.name.clone(),
                args: args.iter().map(|a| self.lower_expr(a)).collect(),
            },
            ast::ExprKind::Field { expr: inner, field } => ExprKind::Field {
                expr: Box::new(self.lower_expr(inner)),
                field: field.name.clone(),
            },
            ast::ExprKind::Index { expr: inner, index } => ExprKind::Index {
                expr: Box::new(self.lower_expr(inner)),
                index: Box::new(self.lower_expr(index)),
            },
            ast::ExprKind::TupleIndex { expr: inner, index } => ExprKind::Field {
                expr: Box::new(self.lower_expr(inner)),
                field: index.to_string(),
            },
            ast::ExprKind::Array(arr) => match arr {
                ast::ArrayExpr::List(exprs) => {
                    ExprKind::Array(exprs.iter().map(|e| self.lower_expr(e)).collect())
                }
                ast::ArrayExpr::Repeat { value, count } => ExprKind::Repeat {
                    elem: Box::new(self.lower_expr(value)),
                    count: Box::new(self.lower_expr(count)),
                },
            },
            ast::ExprKind::Tuple(exprs) => {
                ExprKind::Tuple(exprs.iter().map(|e| self.lower_expr(e)).collect())
            }
            ast::ExprKind::Struct { path, fields, rest } => ExprKind::Struct {
                path: self.lower_path(path),
                fields: fields.iter().map(|f| {
                    (f.name.name.clone(), self.lower_expr(&f.value))
                }).collect(),
                rest: rest.as_ref().map(|r| Box::new(self.lower_expr(r))),
            },
            ast::ExprKind::If { cond, then_branch, else_branch } => ExprKind::If {
                cond: Box::new(self.lower_expr(cond)),
                then_branch: Box::new(self.lower_block(then_branch)),
                else_branch: else_branch.as_ref().map(|e| Box::new(self.lower_expr(e))),
            },
            ast::ExprKind::Match { expr: scrutinee, arms } => ExprKind::Match {
                expr: Box::new(self.lower_expr(scrutinee)),
                arms: arms.iter().map(|a| Arm {
                    pattern: self.lower_pattern(&a.pattern),
                    guard: a.guard.as_ref().map(|g| Box::new(self.lower_expr(g))),
                    body: Box::new(self.lower_expr(&a.body)),
                }).collect(),
            },
            ast::ExprKind::Loop { label, body } => ExprKind::Loop {
                body: Box::new(self.lower_block(body)),
                label: label.as_ref().map(|l| l.name.clone()),
            },
            ast::ExprKind::While { label, cond, body } => ExprKind::While {
                cond: Box::new(self.lower_expr(cond)),
                body: Box::new(self.lower_block(body)),
                label: label.as_ref().map(|l| l.name.clone()),
            },
            ast::ExprKind::For { label, pat, iter, body } => ExprKind::For {
                pattern: self.lower_pattern(pat),
                iter: Box::new(self.lower_expr(iter)),
                body: Box::new(self.lower_block(body)),
                label: label.as_ref().map(|l| l.name.clone()),
            },
            ast::ExprKind::Block(block) => ExprKind::Block(Box::new(self.lower_block(block))),
            ast::ExprKind::Closure { is_move, params, ret_type: _, body } => ExprKind::Closure {
                params: params.iter().map(|p| {
                    (self.lower_pattern(&p.pattern), p.ty.as_ref().map(|t| self.lower_type(t)))
                }).collect(),
                body: Box::new(self.lower_expr(body)),
                is_async: false,
                is_move: *is_move,
            },
            ast::ExprKind::Return(inner) => {
                ExprKind::Return(inner.as_ref().map(|e| Box::new(self.lower_expr(e))))
            }
            ast::ExprKind::Break { label, expr: inner } => ExprKind::Break {
                label: label.as_ref().map(|l| l.name.clone()),
                value: inner.as_ref().map(|e| Box::new(self.lower_expr(e))),
            },
            ast::ExprKind::Continue(label) => {
                ExprKind::Continue(label.as_ref().map(|l| l.name.clone()))
            }
            ast::ExprKind::Assign { target, value } => ExprKind::Assign {
                lhs: Box::new(self.lower_expr(target)),
                rhs: Box::new(self.lower_expr(value)),
            },
            ast::ExprKind::AssignOp { op, target, value } => ExprKind::AssignOp {
                op: self.lower_binop(*op),
                lhs: Box::new(self.lower_expr(target)),
                rhs: Box::new(self.lower_expr(value)),
            },
            ast::ExprKind::Range { start, end, inclusive } => ExprKind::Range {
                lo: start.as_ref().map(|e| Box::new(self.lower_expr(e))),
                hi: end.as_ref().map(|e| Box::new(self.lower_expr(e))),
                inclusive: *inclusive,
            },
            ast::ExprKind::Try(inner) => ExprKind::Try(Box::new(self.lower_expr(inner))),
            ast::ExprKind::Await(inner) => ExprKind::Await(Box::new(self.lower_expr(inner))),
            ast::ExprKind::Ref { mutable, expr: inner } => ExprKind::Ref {
                mutable: *mutable,
                expr: Box::new(self.lower_expr(inner)),
            },
            ast::ExprKind::Deref(inner) => ExprKind::Deref(Box::new(self.lower_expr(inner))),
            ast::ExprKind::Cast { expr: inner, ty } => ExprKind::Cast {
                expr: Box::new(self.lower_expr(inner)),
                ty: self.lower_type(ty),
            },
            ast::ExprKind::TypeAscription { expr: inner, ty: _ } => {
                // Ignore type ascription, just lower the expression
                return self.lower_expr(inner);
            }
            ast::ExprKind::Let { pat, expr: inner } => {
                // if let expression condition
                ExprKind::Err // TODO: Handle let expressions in conditions
            }
            ast::ExprKind::MacroCall { path: _, args: _ } => {
                // TODO: Macro expansion
                ExprKind::Err
            }
            ast::ExprKind::Unsafe(block) => ExprKind::Block(Box::new(self.lower_block(block))),
            ast::ExprKind::Error => ExprKind::Err,
        };
        Expr::new(kind, span)
    }

    fn lower_literal(&self, lit: &ast::Lit) -> Literal {
        match lit {
            ast::Lit::Int(v, suffix) => {
                let ty = suffix.as_ref().and_then(|s| match s.as_str() {
                    "i8" => Some(IntType::I8),
                    "i16" => Some(IntType::I16),
                    "i32" => Some(IntType::I32),
                    "i64" => Some(IntType::I64),
                    "i128" => Some(IntType::I128),
                    "isize" => Some(IntType::Isize),
                    _ => None,
                });
                if ty.is_some() {
                    Literal::Int(*v, ty)
                } else {
                    // Check for unsigned suffix
                    let uty = suffix.as_ref().and_then(|s| match s.as_str() {
                        "u8" => Some(UintType::U8),
                        "u16" => Some(UintType::U16),
                        "u32" => Some(UintType::U32),
                        "u64" => Some(UintType::U64),
                        "u128" => Some(UintType::U128),
                        "usize" => Some(UintType::Usize),
                        _ => None,
                    });
                    if uty.is_some() {
                        Literal::Uint(*v as u128, uty)
                    } else {
                        Literal::Int(*v, None)
                    }
                }
            }
            ast::Lit::Float(v, suffix) => {
                let ty = suffix.as_ref().and_then(|s| match s.as_str() {
                    "f32" => Some(FloatType::F32),
                    "f64" => Some(FloatType::F64),
                    _ => None,
                });
                Literal::Float(*v, ty)
            }
            ast::Lit::Str(s) => Literal::Str(s.clone()),
            ast::Lit::ByteStr(b) => Literal::ByteStr(b.clone()),
            ast::Lit::Char(c) => Literal::Char(*c),
            ast::Lit::Byte(b) => Literal::Uint(*b as u128, Some(UintType::U8)),
            ast::Lit::Bool(b) => Literal::Bool(*b),
        }
    }

    fn lower_binop(&self, op: ast::BinOp) -> BinaryOp {
        match op {
            ast::BinOp::Add => BinaryOp::Add,
            ast::BinOp::Sub => BinaryOp::Sub,
            ast::BinOp::Mul => BinaryOp::Mul,
            ast::BinOp::Div => BinaryOp::Div,
            ast::BinOp::Rem => BinaryOp::Rem,
            ast::BinOp::And => BinaryOp::And,
            ast::BinOp::Or => BinaryOp::Or,
            ast::BinOp::BitAnd => BinaryOp::BitAnd,
            ast::BinOp::BitOr => BinaryOp::BitOr,
            ast::BinOp::BitXor => BinaryOp::BitXor,
            ast::BinOp::Shl => BinaryOp::Shl,
            ast::BinOp::Shr => BinaryOp::Shr,
            ast::BinOp::Eq => BinaryOp::Eq,
            ast::BinOp::Ne => BinaryOp::Ne,
            ast::BinOp::Lt => BinaryOp::Lt,
            ast::BinOp::Le => BinaryOp::Le,
            ast::BinOp::Gt => BinaryOp::Gt,
            ast::BinOp::Ge => BinaryOp::Ge,
        }
    }

    fn lower_unaryop(&self, op: ast::UnaryOp) -> UnaryOp {
        match op {
            ast::UnaryOp::Neg => UnaryOp::Neg,
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::Deref => UnaryOp::Neg, // Deref handled separately
            ast::UnaryOp::Ref => UnaryOp::Neg,   // Ref handled separately
            ast::UnaryOp::RefMut => UnaryOp::Neg, // RefMut handled separately
        }
    }

    // ========================================================================
    // Pattern lowering
    // ========================================================================

    fn lower_pattern(&self, pat: &ast::Pattern) -> Pattern {
        let kind = match &pat.kind {
            ast::PatternKind::Wild => PatternKind::Wild,
            ast::PatternKind::Ident { mutable, by_ref: _, name, subpat } => PatternKind::Ident {
                mutable: *mutable,
                name: name.name.clone(),
                binding: subpat.as_ref().map(|p| Box::new(self.lower_pattern(p))),
            },
            ast::PatternKind::Lit(lit) => PatternKind::Lit(self.lower_literal(lit)),
            ast::PatternKind::Tuple(pats) => {
                PatternKind::Tuple(pats.iter().map(|p| self.lower_pattern(p)).collect())
            }
            ast::PatternKind::Struct { path, fields, rest } => PatternKind::Struct {
                path: self.lower_path(path),
                fields: fields.iter().map(|f| FieldPattern {
                    name: f.name.name.clone(),
                    pattern: self.lower_pattern(&f.pattern),
                }).collect(),
                rest: *rest,
            },
            ast::PatternKind::TupleStruct { path, fields } => PatternKind::TupleStruct {
                path: self.lower_path(path),
                elems: fields.iter().map(|p| self.lower_pattern(p)).collect(),
            },
            ast::PatternKind::Slice(pats) => {
                PatternKind::Tuple(pats.iter().map(|p| self.lower_pattern(p)).collect())
            }
            ast::PatternKind::Or(pats) => {
                PatternKind::Or(pats.iter().map(|p| self.lower_pattern(p)).collect())
            }
            ast::PatternKind::Range { start: _, end: _, inclusive: _ } => {
                // TODO: Proper range pattern
                PatternKind::Wild
            }
            ast::PatternKind::Ref { mutable, inner } => PatternKind::Ref {
                mutable: *mutable,
                pattern: Box::new(self.lower_pattern(inner)),
            },
            ast::PatternKind::Path(path) => {
                // Single-segment lowercase path in pattern position is likely a binding
                // e.g., `val` in `Some(val)` should be a binding, not a path reference
                if path.segments.len() == 1 && path.segments[0].args.is_none() {
                    let name = &path.segments[0].ident.name;
                    // Lowercase names are bindings, uppercase names are enum variants/constants
                    if name.chars().next().map(|c| c.is_lowercase() || c == '_').unwrap_or(false) {
                        PatternKind::Ident {
                            mutable: false,
                            name: name.clone(),
                            binding: None,
                        }
                    } else {
                        PatternKind::Path(self.lower_path(path))
                    }
                } else {
                    PatternKind::Path(self.lower_path(path))
                }
            }
        };
        Pattern { kind, span: self.convert_span(pat.span) }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    fn qualified_name(&self, name: &str) -> String {
        if self.module_prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}::{}", self.module_prefix, name)
        }
    }

    fn pattern_name(&self, pat: &ast::Pattern) -> String {
        match &pat.kind {
            ast::PatternKind::Ident { name, .. } => name.name.clone(),
            ast::PatternKind::Wild => "_".to_string(),
            // Handle Path patterns - single-segment paths are used for simple parameter names
            ast::PatternKind::Path(path) if path.segments.len() == 1 => {
                path.segments[0].ident.name.clone()
            },
            _ => "_".to_string(),
        }
    }

    fn type_name(&self, ty: &ast::Type) -> String {
        match &ty.kind {
            ast::TypeKind::Path(path) => self.path_to_string(path),
            _ => "Unknown".to_string(),
        }
    }

    fn path_to_string(&self, path: &ast::Path) -> String {
        path.segments.iter()
            .map(|s| s.ident.name.as_str())
            .collect::<Vec<_>>()
            .join("::")
    }
}

/// Parse and lower source code to HIR using parser2
pub fn parse_and_lower(source: &str, crate_name: &str) -> Result<Crate, Vec<String>> {
    let (ast, errors) = crate::parser2::parse(source);

    if !errors.is_empty() {
        let error_msgs: Vec<String> = errors.iter()
            .map(|e| format!("{:?}", e))
            .collect();
        return Err(error_msgs);
    }

    match ast {
        Some(source_file) => {
            let mut ctx = LowerContext::new(crate_name.to_string());
            ctx.lower_source_file(&source_file);
            Ok(ctx.hir_crate)
        }
        None => Err(vec!["Failed to parse source file".to_string()]),
    }
}
