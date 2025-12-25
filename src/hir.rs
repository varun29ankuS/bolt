//! High-level Intermediate Representation (HIR)
//!
//! This is a simplified AST that's easier to analyze than raw syn AST.
//! It's fully resolved (no more name resolution needed) and typed.
//!
//! Supports expression-level incremental compilation via content hashing.

use crate::error::Span;
use indexmap::IndexMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub type HirId = u32;
pub type DefId = u32;
pub type TypeId = u32;

#[derive(Debug)]
pub struct Crate {
    pub name: String,
    pub items: IndexMap<DefId, Item>,
    pub entry_point: Option<DefId>,
    /// Import aliases: short_name -> full_path (e.g., "add" -> "math::add")
    pub imports: std::collections::HashMap<String, String>,
    /// Macro definitions: name -> definition
    pub macros: std::collections::HashMap<String, MacroDef>,
}

impl Crate {
    pub fn new(name: String) -> Self {
        Self {
            name,
            items: IndexMap::new(),
            entry_point: None,
            imports: std::collections::HashMap::new(),
            macros: std::collections::HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct Item {
    pub id: DefId,
    pub name: String,
    pub kind: ItemKind,
    pub visibility: Visibility,
    pub span: Span,
}

#[derive(Debug)]
pub enum ItemKind {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    Const(Const),
    Static(Static),
    Impl(Impl),
    Trait(Trait),
    TypeAlias(TypeAlias),
    Module(Module),
    Macro(MacroDef),
}

/// A macro_rules! definition
#[derive(Debug, Clone)]
pub struct MacroDef {
    pub rules: Vec<MacroRule>,
}

/// A single rule in a macro_rules! definition
#[derive(Debug, Clone)]
pub struct MacroRule {
    /// The pattern to match (as token trees)
    pub pattern: Vec<MacroToken>,
    /// The expansion template (as token trees)
    pub template: Vec<MacroToken>,
}

/// Token in a macro pattern or template
#[derive(Debug, Clone)]
pub enum MacroToken {
    /// Literal identifier
    Ident(String),
    /// Literal punctuation
    Punct(char),
    /// Metavariable: $name:kind
    MetaVar { name: String, kind: MetaVarKind },
    /// Repetition: $(...)*  or  $(...)+ or  $(...),*
    Repetition {
        tokens: Vec<MacroToken>,
        separator: Option<char>,
        kind: RepetitionKind,
    },
    /// Grouped tokens: (...) or [...] or {...}
    Group {
        delimiter: Delimiter,
        tokens: Vec<MacroToken>,
    },
    /// Literal token (number, string, etc)
    Literal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaVarKind {
    Expr,   // $x:expr
    Ty,     // $t:ty
    Ident,  // $i:ident
    Pat,    // $p:pat
    Stmt,   // $s:stmt
    Block,  // $b:block
    Item,   // $i:item
    Tt,     // $t:tt (token tree)
    Literal, // $l:literal
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepetitionKind {
    ZeroOrMore,  // *
    OneOrMore,   // +
    ZeroOrOne,   // ?
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
    Paren,   // ()
    Bracket, // []
    Brace,   // {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
    Crate,
    Super,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub sig: FnSig,
    pub body: Option<Block>,
    pub is_async: bool,
    pub is_const: bool,
    pub is_unsafe: bool,
}

#[derive(Debug, Clone)]
pub struct FnSig {
    pub inputs: Vec<(String, Type)>,
    pub output: Type,
    pub generics: Generics,
}

#[derive(Debug, Default, Clone)]
pub struct Generics {
    pub params: Vec<GenericParam>,
    pub where_clause: Vec<WherePredicate>,
}

#[derive(Debug, Clone)]
pub enum GenericParam {
    Type { name: String, bounds: Vec<TypeBound> },
    Lifetime { name: String },
    Const { name: String, ty: Type },
}

#[derive(Debug, Clone)]
pub struct WherePredicate {
    pub ty: Type,
    pub bounds: Vec<TypeBound>,
}

#[derive(Debug, Clone)]
pub enum TypeBound {
    Trait(Path),
    Lifetime(String),
}

#[derive(Debug, Clone)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    Unit,
    Bool,
    Char,
    Int(IntType),
    Uint(UintType),
    Float(FloatType),
    Str,
    Ref { lifetime: Option<String>, mutable: bool, inner: Box<Type> },
    Ptr { mutable: bool, inner: Box<Type> },
    Slice(Box<Type>),
    Array { elem: Box<Type>, len: usize },
    Tuple(Vec<Type>),
    Path(Path),
    Fn { inputs: Vec<Type>, output: Box<Type> },
    Never,
    Infer,
    Error,
    /// impl Trait - existential type that implements a trait
    ImplTrait(Vec<TypeBound>),
    /// dyn Trait - trait object
    DynTrait(Vec<TypeBound>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntType {
    I8, I16, I32, I64, I128, Isize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UintType {
    U8, U16, U32, U64, U128, Usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatType {
    F32, F64,
}

#[derive(Debug, Clone)]
pub struct Path {
    pub segments: Vec<PathSegment>,
}

#[derive(Debug, Clone)]
pub struct PathSegment {
    pub ident: String,
    pub args: Option<GenericArgs>,
}

#[derive(Debug, Clone)]
pub struct GenericArgs {
    pub args: Vec<GenericArg>,
}

#[derive(Debug, Clone)]
pub enum GenericArg {
    Type(Type),
    Lifetime(String),
    Const(Expr),
}

#[derive(Debug)]
pub struct Struct {
    pub generics: Generics,
    pub kind: StructKind,
}

#[derive(Debug)]
pub enum StructKind {
    Unit,
    Tuple(Vec<Type>),
    Named(Vec<Field>),
}

#[derive(Debug)]
pub struct Field {
    pub name: String,
    pub ty: Type,
    pub visibility: Visibility,
}

#[derive(Debug)]
pub struct Enum {
    pub generics: Generics,
    pub variants: Vec<Variant>,
}

#[derive(Debug)]
pub struct Variant {
    pub name: String,
    pub kind: StructKind,
    pub discriminant: Option<Expr>,
}

#[derive(Debug)]
pub struct Const {
    pub ty: Type,
    pub value: Expr,
}

#[derive(Debug)]
pub struct Static {
    pub ty: Type,
    pub value: Expr,
    pub mutable: bool,
}

#[derive(Debug)]
pub struct Impl {
    pub generics: Generics,
    pub trait_ref: Option<Path>,
    pub self_ty: Type,
    pub items: Vec<DefId>,
    /// Associated type bindings: type Item = ConcreteType;
    pub assoc_types: Vec<AssocTypeBinding>,
}

/// An associated type binding in an impl block
#[derive(Debug, Clone)]
pub struct AssocTypeBinding {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug)]
pub struct Trait {
    pub generics: Generics,
    pub bounds: Vec<TypeBound>,
    pub items: Vec<DefId>,
    /// Associated types declared in this trait
    pub assoc_types: Vec<AssocTypeDecl>,
}

/// An associated type declaration in a trait
#[derive(Debug, Clone)]
pub struct AssocTypeDecl {
    pub name: String,
    pub bounds: Vec<TypeBound>,
    pub default: Option<Type>,
    pub span: Span,
}

#[derive(Debug)]
pub struct TypeAlias {
    pub generics: Generics,
    pub ty: Type,
}

#[derive(Debug)]
pub struct Module {
    pub items: Vec<DefId>,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub expr: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    Let { pattern: Pattern, ty: Option<Type>, init: Option<Expr> },
    Expr(Expr),
    Semi(Expr),
    Item(DefId),
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum PatternKind {
    Wild,
    Ident { mutable: bool, name: String, binding: Option<Box<Pattern>> },
    Ref { mutable: bool, pattern: Box<Pattern> },
    Tuple(Vec<Pattern>),
    Struct { path: Path, fields: Vec<FieldPattern>, rest: bool },
    TupleStruct { path: Path, elems: Vec<Pattern> },
    Path(Path),
    Lit(Literal),
    Range { lo: Option<Box<Pattern>>, hi: Option<Box<Pattern>>, inclusive: bool },
    Or(Vec<Pattern>),
}

#[derive(Debug, Clone)]
pub struct FieldPattern {
    pub name: String,
    pub pattern: Pattern,
}

/// HIR expression with content hash for incremental compilation.
///
/// The `content_hash` field is computed from the structural content of the
/// expression (kind + children), allowing cache lookups without re-parsing.
/// Two expressions with identical structure have the same hash, even from
/// different source locations.
#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Option<TypeId>,
    pub span: Span,
    /// Content hash for incremental compilation (hash of kind + children).
    /// Zero if not yet computed.
    pub content_hash: u64,
}

impl Expr {
    /// Create a new expression with automatic content hash computation.
    pub fn new(kind: ExprKind, span: Span) -> Self {
        let content_hash = compute_expr_hash(&kind);
        Self {
            kind,
            ty: None,
            span,
            content_hash,
        }
    }

    /// Create expression without hash (for backward compatibility during lowering).
    pub fn unhashed(kind: ExprKind, span: Span) -> Self {
        Self {
            kind,
            ty: None,
            span,
            content_hash: 0,
        }
    }

    /// Compute and set the content hash if not already computed.
    pub fn ensure_hashed(&mut self) {
        if self.content_hash == 0 {
            self.content_hash = compute_expr_hash(&self.kind);
        }
    }
}

/// Compute content hash for an expression kind.
/// This recursively hashes the structure, not the span or type.
fn compute_expr_hash(kind: &ExprKind) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_expr_kind(kind, &mut hasher);
    hasher.finish()
}

fn hash_expr_kind<H: Hasher>(kind: &ExprKind, hasher: &mut H) {
    std::mem::discriminant(kind).hash(hasher);
    match kind {
        ExprKind::Lit(lit) => hash_literal(lit, hasher),
        ExprKind::Path(path) => hash_path(path, hasher),
        ExprKind::Unary { op, expr } => {
            std::mem::discriminant(op).hash(hasher);
            hash_expr_kind(&expr.kind, hasher);
        }
        ExprKind::Binary { op, lhs, rhs } => {
            std::mem::discriminant(op).hash(hasher);
            hash_expr_kind(&lhs.kind, hasher);
            hash_expr_kind(&rhs.kind, hasher);
        }
        ExprKind::Assign { lhs, rhs } | ExprKind::AssignOp { op: _, lhs, rhs } => {
            hash_expr_kind(&lhs.kind, hasher);
            hash_expr_kind(&rhs.kind, hasher);
        }
        ExprKind::Index { expr, index } => {
            hash_expr_kind(&expr.kind, hasher);
            hash_expr_kind(&index.kind, hasher);
        }
        ExprKind::Field { expr, field } => {
            hash_expr_kind(&expr.kind, hasher);
            field.hash(hasher);
        }
        ExprKind::Call { func, args } => {
            hash_expr_kind(&func.kind, hasher);
            args.len().hash(hasher);
            for arg in args {
                hash_expr_kind(&arg.kind, hasher);
            }
        }
        ExprKind::MethodCall { receiver, method, args } => {
            hash_expr_kind(&receiver.kind, hasher);
            method.hash(hasher);
            args.len().hash(hasher);
            for arg in args {
                hash_expr_kind(&arg.kind, hasher);
            }
        }
        ExprKind::Tuple(exprs) | ExprKind::Array(exprs) => {
            exprs.len().hash(hasher);
            for expr in exprs {
                hash_expr_kind(&expr.kind, hasher);
            }
        }
        ExprKind::Repeat { elem, count } => {
            hash_expr_kind(&elem.kind, hasher);
            hash_expr_kind(&count.kind, hasher);
        }
        ExprKind::Struct { path, fields, rest } => {
            hash_path(path, hasher);
            fields.len().hash(hasher);
            for (name, expr) in fields {
                name.hash(hasher);
                hash_expr_kind(&expr.kind, hasher);
            }
            rest.is_some().hash(hasher);
            if let Some(r) = rest {
                hash_expr_kind(&r.kind, hasher);
            }
        }
        ExprKind::If { cond, then_branch, else_branch } => {
            hash_expr_kind(&cond.kind, hasher);
            hash_block(then_branch, hasher);
            else_branch.is_some().hash(hasher);
            if let Some(eb) = else_branch {
                hash_expr_kind(&eb.kind, hasher);
            }
        }
        ExprKind::IfLet { pattern, expr, then_branch, else_branch } => {
            hash_pattern(&pattern.kind, hasher);
            hash_expr_kind(&expr.kind, hasher);
            hash_block(then_branch, hasher);
            else_branch.is_some().hash(hasher);
            if let Some(eb) = else_branch {
                hash_expr_kind(&eb.kind, hasher);
            }
        }
        ExprKind::Match { expr, arms } => {
            hash_expr_kind(&expr.kind, hasher);
            arms.len().hash(hasher);
            for arm in arms {
                hash_pattern(&arm.pattern.kind, hasher);
                arm.guard.is_some().hash(hasher);
                if let Some(g) = &arm.guard {
                    hash_expr_kind(&g.kind, hasher);
                }
                hash_expr_kind(&arm.body.kind, hasher);
            }
        }
        ExprKind::Loop { body, label } | ExprKind::While { cond: _, body, label } => {
            hash_block(body, hasher);
            label.hash(hasher);
        }
        ExprKind::WhileLet { pattern, expr, body, label } => {
            hash_pattern(&pattern.kind, hasher);
            hash_expr_kind(&expr.kind, hasher);
            hash_block(body, hasher);
            label.hash(hasher);
        }
        ExprKind::For { pattern, iter, body, label } => {
            hash_pattern(&pattern.kind, hasher);
            hash_expr_kind(&iter.kind, hasher);
            hash_block(body, hasher);
            label.hash(hasher);
        }
        ExprKind::Block(block) => hash_block(block, hasher),
        ExprKind::Return(opt_expr) => {
            opt_expr.is_some().hash(hasher);
            if let Some(e) = opt_expr {
                hash_expr_kind(&e.kind, hasher);
            }
        }
        ExprKind::Break { label, value } => {
            label.hash(hasher);
            value.is_some().hash(hasher);
            if let Some(v) = value {
                hash_expr_kind(&v.kind, hasher);
            }
        }
        ExprKind::Continue(label) => label.hash(hasher),
        ExprKind::Ref { mutable, expr } => {
            mutable.hash(hasher);
            hash_expr_kind(&expr.kind, hasher);
        }
        ExprKind::Deref(expr) => hash_expr_kind(&expr.kind, hasher),
        ExprKind::Cast { expr, ty } => {
            hash_expr_kind(&expr.kind, hasher);
            hash_type(&ty.kind, hasher);
        }
        ExprKind::Range { lo, hi, inclusive } => {
            lo.is_some().hash(hasher);
            if let Some(l) = lo {
                hash_expr_kind(&l.kind, hasher);
            }
            hi.is_some().hash(hasher);
            if let Some(h) = hi {
                hash_expr_kind(&h.kind, hasher);
            }
            inclusive.hash(hasher);
        }
        ExprKind::Closure { params, body, is_async, is_move } => {
            params.len().hash(hasher);
            for (pat, ty) in params {
                hash_pattern(&pat.kind, hasher);
                ty.is_some().hash(hasher);
                if let Some(t) = ty {
                    hash_type(&t.kind, hasher);
                }
            }
            hash_expr_kind(&body.kind, hasher);
            is_async.hash(hasher);
            is_move.hash(hasher);
        }
        ExprKind::Await(expr) | ExprKind::Try(expr) => {
            hash_expr_kind(&expr.kind, hasher);
        }
        ExprKind::Err => {}
    }
}

fn hash_block<H: Hasher>(block: &Block, hasher: &mut H) {
    block.stmts.len().hash(hasher);
    for stmt in &block.stmts {
        hash_stmt(&stmt.kind, hasher);
    }
    block.expr.is_some().hash(hasher);
    if let Some(e) = &block.expr {
        hash_expr_kind(&e.kind, hasher);
    }
}

fn hash_stmt<H: Hasher>(kind: &StmtKind, hasher: &mut H) {
    std::mem::discriminant(kind).hash(hasher);
    match kind {
        StmtKind::Let { pattern, ty, init } => {
            hash_pattern(&pattern.kind, hasher);
            ty.is_some().hash(hasher);
            if let Some(t) = ty {
                hash_type(&t.kind, hasher);
            }
            init.is_some().hash(hasher);
            if let Some(i) = init {
                hash_expr_kind(&i.kind, hasher);
            }
        }
        StmtKind::Expr(e) | StmtKind::Semi(e) => hash_expr_kind(&e.kind, hasher),
        StmtKind::Item(def_id) => def_id.hash(hasher),
    }
}

fn hash_pattern<H: Hasher>(kind: &PatternKind, hasher: &mut H) {
    std::mem::discriminant(kind).hash(hasher);
    match kind {
        PatternKind::Wild => {}
        PatternKind::Ident { mutable, name, binding } => {
            mutable.hash(hasher);
            name.hash(hasher);
            binding.is_some().hash(hasher);
            if let Some(b) = binding {
                hash_pattern(&b.kind, hasher);
            }
        }
        PatternKind::Ref { mutable, pattern } => {
            mutable.hash(hasher);
            hash_pattern(&pattern.kind, hasher);
        }
        PatternKind::Tuple(pats) => {
            pats.len().hash(hasher);
            for p in pats {
                hash_pattern(&p.kind, hasher);
            }
        }
        PatternKind::Struct { path, fields, rest } => {
            hash_path(path, hasher);
            fields.len().hash(hasher);
            for f in fields {
                f.name.hash(hasher);
                hash_pattern(&f.pattern.kind, hasher);
            }
            rest.hash(hasher);
        }
        PatternKind::TupleStruct { path, elems } => {
            hash_path(path, hasher);
            elems.len().hash(hasher);
            for e in elems {
                hash_pattern(&e.kind, hasher);
            }
        }
        PatternKind::Path(path) => hash_path(path, hasher),
        PatternKind::Lit(lit) => hash_literal(lit, hasher),
        PatternKind::Range { lo, hi, inclusive } => {
            lo.is_some().hash(hasher);
            if let Some(l) = lo {
                hash_pattern(&l.kind, hasher);
            }
            hi.is_some().hash(hasher);
            if let Some(h) = hi {
                hash_pattern(&h.kind, hasher);
            }
            inclusive.hash(hasher);
        }
        PatternKind::Or(pats) => {
            pats.len().hash(hasher);
            for p in pats {
                hash_pattern(&p.kind, hasher);
            }
        }
    }
}

fn hash_literal<H: Hasher>(lit: &Literal, hasher: &mut H) {
    std::mem::discriminant(lit).hash(hasher);
    match lit {
        Literal::Bool(b) => b.hash(hasher),
        Literal::Char(c) => c.hash(hasher),
        Literal::Int(v, ty) => {
            v.hash(hasher);
            ty.hash(hasher);
        }
        Literal::Uint(v, ty) => {
            v.hash(hasher);
            ty.hash(hasher);
        }
        Literal::Float(v, ty) => {
            v.to_bits().hash(hasher);
            ty.hash(hasher);
        }
        Literal::Str(s) => s.hash(hasher),
        Literal::ByteStr(b) => b.hash(hasher),
    }
}

fn hash_path<H: Hasher>(path: &Path, hasher: &mut H) {
    path.segments.len().hash(hasher);
    for seg in &path.segments {
        seg.ident.hash(hasher);
        seg.args.is_some().hash(hasher);
        if let Some(args) = &seg.args {
            args.args.len().hash(hasher);
            for arg in &args.args {
                hash_generic_arg(arg, hasher);
            }
        }
    }
}

fn hash_generic_arg<H: Hasher>(arg: &GenericArg, hasher: &mut H) {
    std::mem::discriminant(arg).hash(hasher);
    match arg {
        GenericArg::Type(ty) => hash_type(&ty.kind, hasher),
        GenericArg::Lifetime(lt) => lt.hash(hasher),
        GenericArg::Const(e) => hash_expr_kind(&e.kind, hasher),
    }
}

fn hash_type<H: Hasher>(kind: &TypeKind, hasher: &mut H) {
    std::mem::discriminant(kind).hash(hasher);
    match kind {
        TypeKind::Unit | TypeKind::Bool | TypeKind::Char | TypeKind::Str
        | TypeKind::Never | TypeKind::Infer | TypeKind::Error => {}
        TypeKind::Int(t) => t.hash(hasher),
        TypeKind::Uint(t) => t.hash(hasher),
        TypeKind::Float(t) => t.hash(hasher),
        TypeKind::Ref { lifetime, mutable, inner } => {
            lifetime.hash(hasher);
            mutable.hash(hasher);
            hash_type(&inner.kind, hasher);
        }
        TypeKind::Ptr { mutable, inner } => {
            mutable.hash(hasher);
            hash_type(&inner.kind, hasher);
        }
        TypeKind::Slice(inner) => hash_type(&inner.kind, hasher),
        TypeKind::Array { elem, len } => {
            hash_type(&elem.kind, hasher);
            len.hash(hasher);
        }
        TypeKind::Tuple(tys) => {
            tys.len().hash(hasher);
            for t in tys {
                hash_type(&t.kind, hasher);
            }
        }
        TypeKind::Path(path) => hash_path(path, hasher),
        TypeKind::Fn { inputs, output } => {
            inputs.len().hash(hasher);
            for i in inputs {
                hash_type(&i.kind, hasher);
            }
            hash_type(&output.kind, hasher);
        }
        TypeKind::ImplTrait(bounds) | TypeKind::DynTrait(bounds) => {
            bounds.len().hash(hasher);
            for bound in bounds {
                hash_type_bound(bound, hasher);
            }
        }
    }
}

fn hash_type_bound<H: Hasher>(bound: &TypeBound, hasher: &mut H) {
    match bound {
        TypeBound::Trait(path) => {
            0u8.hash(hasher);
            hash_path(path, hasher);
        }
        TypeBound::Lifetime(lt) => {
            1u8.hash(hasher);
            lt.hash(hasher);
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Lit(Literal),
    Path(Path),
    Unary { op: UnaryOp, expr: Box<Expr> },
    Binary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Assign { lhs: Box<Expr>, rhs: Box<Expr> },
    AssignOp { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Index { expr: Box<Expr>, index: Box<Expr> },
    Field { expr: Box<Expr>, field: String },
    Call { func: Box<Expr>, args: Vec<Expr> },
    MethodCall { receiver: Box<Expr>, method: String, args: Vec<Expr> },
    Tuple(Vec<Expr>),
    Array(Vec<Expr>),
    Repeat { elem: Box<Expr>, count: Box<Expr> },
    Struct { path: Path, fields: Vec<(String, Expr)>, rest: Option<Box<Expr>> },
    If { cond: Box<Expr>, then_branch: Box<Block>, else_branch: Option<Box<Expr>> },
    IfLet { pattern: Pattern, expr: Box<Expr>, then_branch: Box<Block>, else_branch: Option<Box<Expr>> },
    Match { expr: Box<Expr>, arms: Vec<Arm> },
    Loop { body: Box<Block>, label: Option<String> },
    While { cond: Box<Expr>, body: Box<Block>, label: Option<String> },
    WhileLet { pattern: Pattern, expr: Box<Expr>, body: Box<Block>, label: Option<String> },
    For { pattern: Pattern, iter: Box<Expr>, body: Box<Block>, label: Option<String> },
    Block(Box<Block>),
    Return(Option<Box<Expr>>),
    Break { label: Option<String>, value: Option<Box<Expr>> },
    Continue(Option<String>),
    Ref { mutable: bool, expr: Box<Expr> },
    Deref(Box<Expr>),
    Cast { expr: Box<Expr>, ty: Type },
    Range { lo: Option<Box<Expr>>, hi: Option<Box<Expr>>, inclusive: bool },
    Closure { params: Vec<(Pattern, Option<Type>)>, body: Box<Expr>, is_async: bool, is_move: bool },
    Await(Box<Expr>),
    Try(Box<Expr>),
    Err,
}

#[derive(Debug, Clone)]
pub struct Arm {
    pub pattern: Pattern,
    pub guard: Option<Box<Expr>>,
    pub body: Box<Expr>,
}

#[derive(Debug, Clone)]
pub enum Literal {
    Bool(bool),
    Char(char),
    Int(i128, Option<IntType>),
    Uint(u128, Option<UintType>),
    Float(f64, Option<FloatType>),
    Str(String),
    ByteStr(Vec<u8>),
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Rem,
    And, Or, BitAnd, BitOr, BitXor,
    Shl, Shr,
    Eq, Ne, Lt, Le, Gt, Ge,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;

    #[test]
    fn test_expr_hash_consistency() {
        // Same expression should produce same hash
        let span = Span::dummy();
        let e1 = Expr::new(ExprKind::Lit(Literal::Int(42, None)), span);
        let e2 = Expr::new(ExprKind::Lit(Literal::Int(42, None)), span);
        assert_eq!(e1.content_hash, e2.content_hash);
        assert_ne!(e1.content_hash, 0);
    }

    #[test]
    fn test_expr_hash_differs() {
        let span = Span::dummy();
        let e1 = Expr::new(ExprKind::Lit(Literal::Int(42, None)), span);
        let e2 = Expr::new(ExprKind::Lit(Literal::Int(43, None)), span);
        assert_ne!(e1.content_hash, e2.content_hash);
    }

    #[test]
    fn test_expr_hash_span_independent() {
        // Same expression at different spans should have same hash
        let span1 = Span::new(1, 0, 10);
        let span2 = Span::new(2, 100, 200);
        let e1 = Expr::new(ExprKind::Lit(Literal::Bool(true)), span1);
        let e2 = Expr::new(ExprKind::Lit(Literal::Bool(true)), span2);
        assert_eq!(e1.content_hash, e2.content_hash);
    }

    #[test]
    fn test_unhashed_expr() {
        let span = Span::dummy();
        let mut e = Expr::unhashed(ExprKind::Lit(Literal::Int(42, None)), span);
        assert_eq!(e.content_hash, 0);

        e.ensure_hashed();
        assert_ne!(e.content_hash, 0);
    }

    #[test]
    fn test_binary_expr_hash() {
        let span = Span::dummy();
        let lhs = Box::new(Expr::new(ExprKind::Lit(Literal::Int(1, None)), span));
        let rhs = Box::new(Expr::new(ExprKind::Lit(Literal::Int(2, None)), span));
        let e1 = Expr::new(ExprKind::Binary { op: BinaryOp::Add, lhs: lhs.clone(), rhs: rhs.clone() }, span);

        let lhs2 = Box::new(Expr::new(ExprKind::Lit(Literal::Int(1, None)), span));
        let rhs2 = Box::new(Expr::new(ExprKind::Lit(Literal::Int(2, None)), span));
        let e2 = Expr::new(ExprKind::Binary { op: BinaryOp::Add, lhs: lhs2, rhs: rhs2 }, span);

        assert_eq!(e1.content_hash, e2.content_hash);

        // Different operator should give different hash
        let lhs3 = Box::new(Expr::new(ExprKind::Lit(Literal::Int(1, None)), span));
        let rhs3 = Box::new(Expr::new(ExprKind::Lit(Literal::Int(2, None)), span));
        let e3 = Expr::new(ExprKind::Binary { op: BinaryOp::Sub, lhs: lhs3, rhs: rhs3 }, span);
        assert_ne!(e1.content_hash, e3.content_hash);
    }
}
