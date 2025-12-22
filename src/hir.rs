//! High-level Intermediate Representation (HIR)
//!
//! This is a simplified AST that's easier to analyze than raw syn AST.
//! It's fully resolved (no more name resolution needed) and typed.

use crate::error::Span;
use indexmap::IndexMap;

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
}

impl Crate {
    pub fn new(name: String) -> Self {
        Self {
            name,
            items: IndexMap::new(),
            entry_point: None,
            imports: std::collections::HashMap::new(),
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
}

#[derive(Debug)]
pub struct Trait {
    pub generics: Generics,
    pub bounds: Vec<TypeBound>,
    pub items: Vec<DefId>,
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

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Option<TypeId>,
    pub span: Span,
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
