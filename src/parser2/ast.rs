//! Abstract Syntax Tree for Bolt
//!
//! This AST is produced by the Chumsky parser and lowered to HIR.
//! Designed for clarity and ease of use by LLMs.

use crate::lexer::Span;

/// A complete source file
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub items: Vec<Item>,
    pub span: Span,
}

/// Top-level items
#[derive(Debug, Clone)]
pub struct Item {
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ItemKind {
    /// fn name(args) -> ret { body }
    Function(Function),
    /// struct Name { fields }
    Struct(Struct),
    /// enum Name { variants }
    Enum(Enum),
    /// impl Type { methods } or impl Trait for Type { methods }
    Impl(Impl),
    /// trait Name { methods }
    Trait(Trait),
    /// type Name = Type;
    TypeAlias(TypeAlias),
    /// const NAME: Type = expr;
    Const(Const),
    /// static NAME: Type = expr;
    Static(Static),
    /// mod name { items } or mod name;
    Module(Module),
    /// use path::to::item;
    Use(Use),
    /// macro_rules! name { rules }
    MacroRules(MacroRules),
    /// extern "C" { functions }
    ExternBlock(ExternBlock),
}

/// Function definition
#[derive(Debug, Clone)]
pub struct Function {
    pub name: Ident,
    pub generics: Generics,
    pub params: Vec<Param>,
    pub ret_type: Option<Type>,
    pub body: Option<Block>,
    pub is_async: bool,
    pub is_unsafe: bool,
    pub is_pub: bool,
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct Param {
    pub pattern: Pattern,
    pub ty: Type,
    pub span: Span,
}

/// Struct definition
#[derive(Debug, Clone)]
pub struct Struct {
    pub name: Ident,
    pub generics: Generics,
    pub fields: StructFields,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub enum StructFields {
    Named(Vec<Field>),
    Tuple(Vec<Type>),
    Unit,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: Ident,
    pub ty: Type,
    pub is_pub: bool,
    pub span: Span,
}

/// Enum definition
#[derive(Debug, Clone)]
pub struct Enum {
    pub name: Ident,
    pub generics: Generics,
    pub variants: Vec<Variant>,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub name: Ident,
    pub fields: StructFields,
    pub discriminant: Option<Expr>,
    pub span: Span,
}

/// Impl block
#[derive(Debug, Clone)]
pub struct Impl {
    pub generics: Generics,
    pub trait_: Option<Path>,
    pub self_ty: Type,
    pub items: Vec<ImplItem>,
}

#[derive(Debug, Clone)]
pub struct ImplItem {
    pub kind: ImplItemKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ImplItemKind {
    Function(Function),
    Const(Const),
    TypeAlias(TypeAlias),
}

/// Trait definition
#[derive(Debug, Clone)]
pub struct Trait {
    pub name: Ident,
    pub generics: Generics,
    pub bounds: Vec<TypeBound>,
    pub items: Vec<TraitItem>,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub struct TraitItem {
    pub kind: TraitItemKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TraitItemKind {
    Function(Function),
    Const(Const),
    TypeAlias(TypeAlias),
}

/// Type alias
#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub name: Ident,
    pub generics: Generics,
    pub ty: Option<Type>,
    pub is_pub: bool,
}

/// Const item
#[derive(Debug, Clone)]
pub struct Const {
    pub name: Ident,
    pub ty: Type,
    pub value: Option<Expr>,
    pub is_pub: bool,
}

/// Static item
#[derive(Debug, Clone)]
pub struct Static {
    pub name: Ident,
    pub ty: Type,
    pub value: Option<Expr>,
    pub is_mut: bool,
    pub is_pub: bool,
}

/// Module
#[derive(Debug, Clone)]
pub struct Module {
    pub name: Ident,
    pub items: Option<Vec<Item>>,
    pub is_pub: bool,
}

/// Use statement
#[derive(Debug, Clone)]
pub struct Use {
    pub tree: UseTree,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub enum UseTree {
    Path(Path, Option<Box<UseTree>>),
    Glob,
    Group(Vec<UseTree>),
    Alias(Ident, Ident),
}

/// Macro rules
#[derive(Debug, Clone)]
pub struct MacroRules {
    pub name: Ident,
    pub rules: Vec<MacroRule>,
}

/// extern "C" { fn declarations }
#[derive(Debug, Clone)]
pub struct ExternBlock {
    pub abi: Option<String>,
    pub items: Vec<ExternItem>,
}

/// Item inside extern block
#[derive(Debug, Clone)]
pub struct ExternItem {
    pub name: Ident,
    pub params: Vec<Type>,
    pub ret_type: Option<Type>,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub struct MacroRule {
    pub pattern: Vec<MacroToken>,
    pub body: Vec<MacroToken>,
}

#[derive(Debug, Clone)]
pub enum MacroToken {
    Token(String),
    Var(String, Option<String>), // $name:kind
    Repeat(Vec<MacroToken>, Option<char>, RepeatKind),
}

#[derive(Debug, Clone, Copy)]
pub enum RepeatKind {
    ZeroOrMore, // *
    OneOrMore,  // +
    ZeroOrOne,  // ?
}

/// Generics
#[derive(Debug, Clone, Default)]
pub struct Generics {
    pub params: Vec<GenericParam>,
    pub where_clause: Option<WhereClause>,
}

#[derive(Debug, Clone)]
pub enum GenericParam {
    Type { name: Ident, bounds: Vec<TypeBound>, default: Option<Type> },
    Lifetime { name: Ident, bounds: Vec<Ident> },
    Const { name: Ident, ty: Type, default: Option<Expr> },
}

#[derive(Debug, Clone)]
pub struct WhereClause {
    pub predicates: Vec<WherePredicate>,
}

#[derive(Debug, Clone)]
pub enum WherePredicate {
    Type { ty: Type, bounds: Vec<TypeBound> },
    Lifetime { lifetime: Ident, bounds: Vec<Ident> },
}

#[derive(Debug, Clone)]
pub enum TypeBound {
    Trait(Path),
    Lifetime(Ident),
}

/// Types
#[derive(Debug, Clone)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    /// Named type: Foo, Vec<T>, std::io::Result
    Path(Path),
    /// Reference: &T, &'a mut T
    Ref { lifetime: Option<Ident>, mutable: bool, inner: Box<Type> },
    /// Pointer: *const T, *mut T
    Ptr { mutable: bool, inner: Box<Type> },
    /// Slice: [T]
    Slice(Box<Type>),
    /// Array: [T; N]
    Array(Box<Type>, Box<Expr>),
    /// Tuple: (A, B, C)
    Tuple(Vec<Type>),
    /// Function: fn(A, B) -> C
    Fn { params: Vec<Type>, ret: Box<Type> },
    /// Never: !
    Never,
    /// Inferred: _
    Infer,
    /// impl Trait
    ImplTrait(Vec<TypeBound>),
    /// dyn Trait
    DynTrait(Vec<TypeBound>),
}

/// Path like std::collections::HashMap
#[derive(Debug, Clone)]
pub struct Path {
    pub segments: Vec<PathSegment>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct PathSegment {
    pub ident: Ident,
    pub args: Option<GenericArgs>,
}

#[derive(Debug, Clone)]
pub struct GenericArgs {
    pub args: Vec<GenericArg>,
}

#[derive(Debug, Clone)]
pub enum GenericArg {
    Type(Type),
    Lifetime(Ident),
    Const(Expr),
    /// Associated type binding: `Error = MyError` in `Trait<Error = MyError>`
    Binding(Ident, Type),
}

/// Patterns
#[derive(Debug, Clone)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum PatternKind {
    /// Wildcard: _
    Wild,
    /// Identifier: x, mut x, ref x
    Ident { mutable: bool, by_ref: bool, name: Ident, subpat: Option<Box<Pattern>> },
    /// Literal: 42, "hello", true
    Lit(Lit),
    /// Tuple: (a, b, c)
    Tuple(Vec<Pattern>),
    /// Struct: Foo { x, y: z, .. }
    Struct { path: Path, fields: Vec<FieldPat>, rest: bool },
    /// Tuple struct: Foo(a, b)
    TupleStruct { path: Path, fields: Vec<Pattern> },
    /// Slice: [a, b, .., c]
    Slice(Vec<Pattern>),
    /// Or: a | b | c
    Or(Vec<Pattern>),
    /// Range: 1..=10
    Range { start: Option<Box<Expr>>, end: Option<Box<Expr>>, inclusive: bool },
    /// Reference: &x, &mut x
    Ref { mutable: bool, inner: Box<Pattern> },
    /// Path: None, Some
    Path(Path),
}

#[derive(Debug, Clone)]
pub struct FieldPat {
    pub name: Ident,
    pub pattern: Pattern,
    pub shorthand: bool,
}

/// Expressions
#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Literal: 42, "hello", true
    Lit(Lit),
    /// Path: foo, std::io::Result
    Path(Path),
    /// Binary: a + b
    Binary { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    /// Unary: -x, !x, *x, &x
    Unary { op: UnaryOp, expr: Box<Expr> },
    /// Call: foo(a, b)
    Call { func: Box<Expr>, args: Vec<Expr> },
    /// Method call: x.foo(a, b)
    MethodCall { receiver: Box<Expr>, method: Ident, turbofish: Option<GenericArgs>, args: Vec<Expr> },
    /// Field access: x.foo
    Field { expr: Box<Expr>, field: Ident },
    /// Index: x[i]
    Index { expr: Box<Expr>, index: Box<Expr> },
    /// Tuple index: x.0
    TupleIndex { expr: Box<Expr>, index: u32 },
    /// Array: [a, b, c] or [x; n]
    Array(ArrayExpr),
    /// Tuple: (a, b, c)
    Tuple(Vec<Expr>),
    /// Struct: Foo { x: 1, y: 2 }
    Struct { path: Path, fields: Vec<FieldExpr>, rest: Option<Box<Expr>> },
    /// If: if cond { then } else { else }
    If { cond: Box<Expr>, then_branch: Block, else_branch: Option<Box<Expr>> },
    /// Match: match expr { arms }
    Match { expr: Box<Expr>, arms: Vec<MatchArm> },
    /// Loop: loop { body }
    Loop { label: Option<Ident>, body: Block },
    /// While: while cond { body }
    While { label: Option<Ident>, cond: Box<Expr>, body: Block },
    /// For: for pat in iter { body }
    For { label: Option<Ident>, pat: Pattern, iter: Box<Expr>, body: Block },
    /// Block: { stmts }
    Block(Block),
    /// Closure: |args| body or move |args| body
    Closure { is_move: bool, params: Vec<ClosureParam>, ret_type: Option<Type>, body: Box<Expr> },
    /// Return: return expr
    Return(Option<Box<Expr>>),
    /// Break: break 'label expr
    Break { label: Option<Ident>, expr: Option<Box<Expr>> },
    /// Continue: continue 'label
    Continue(Option<Ident>),
    /// Assign: x = y
    Assign { target: Box<Expr>, value: Box<Expr> },
    /// Compound assign: x += y
    AssignOp { op: BinOp, target: Box<Expr>, value: Box<Expr> },
    /// Range: a..b, a..=b, ..b, a..
    Range { start: Option<Box<Expr>>, end: Option<Box<Expr>>, inclusive: bool },
    /// Try: expr?
    Try(Box<Expr>),
    /// Await: expr.await
    Await(Box<Expr>),
    /// Reference: &expr, &mut expr
    Ref { mutable: bool, expr: Box<Expr> },
    /// Dereference: *expr
    Deref(Box<Expr>),
    /// Cast: expr as Type
    Cast { expr: Box<Expr>, ty: Type },
    /// Type ascription: expr: Type
    TypeAscription { expr: Box<Expr>, ty: Type },
    /// Let expression: let pat = expr (in if/while)
    Let { pat: Pattern, expr: Box<Expr> },
    /// Macro call: println!("hello")
    MacroCall { path: Path, args: String },
    /// Unsafe block: unsafe { body }
    Unsafe(Block),
    /// Error placeholder for recovery
    Error,
}

#[derive(Debug, Clone)]
pub enum ArrayExpr {
    List(Vec<Expr>),
    Repeat { value: Box<Expr>, count: Box<Expr> },
}

#[derive(Debug, Clone)]
pub struct FieldExpr {
    pub name: Ident,
    pub value: Expr,
    pub shorthand: bool,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Box<Expr>>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ClosureParam {
    pub pattern: Pattern,
    pub ty: Option<Type>,
}

/// Block of statements
#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

/// Statements
#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    /// let pat: ty = expr;
    Let { pat: Pattern, ty: Option<Type>, init: Option<Expr> },
    /// expr;
    Expr(Expr),
    /// expr (no semicolon - returns value)
    ExprNoSemi(Expr),
    /// Item in block
    Item(Item),
    /// Empty statement (;)
    Empty,
}

/// Literals
#[derive(Debug, Clone)]
pub enum Lit {
    Int(i128, Option<String>),     // 42, 42i64
    Float(f64, Option<String>),    // 3.14, 3.14f32
    Str(String),                   // "hello"
    ByteStr(Vec<u8>),              // b"hello"
    Char(char),                    // 'a'
    Byte(u8),                      // b'a'
    Bool(bool),                    // true, false
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Rem,       // + - * / %
    And, Or,                        // && ||
    BitAnd, BitOr, BitXor,         // & | ^
    Shl, Shr,                       // << >>
    Eq, Ne, Lt, Le, Gt, Ge,        // == != < <= > >=
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,    // -
    Not,    // !
    Deref,  // *
    Ref,    // &
    RefMut, // &mut
}

/// Identifier
#[derive(Debug, Clone)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self { name: name.into(), span }
    }
}
