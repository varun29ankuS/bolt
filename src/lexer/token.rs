//! Token types for the Bolt lexer
//!
//! This module defines all token types that the lexer produces.
//! Designed for hand-written recursive descent parsing.

use std::fmt;
use std::hash::{Hash, Hasher};

/// Source location for error reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    /// Byte offset of start
    pub start: u32,
    /// Byte offset of end (exclusive)
    pub end: u32,
}

impl Span {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    pub fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }

    /// Merge two spans into one covering both
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn len(&self) -> u32 {
        self.end - self.start
    }
}

/// A token with its span
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn is(&self, kind: TokenKind) -> bool {
        self.kind == kind
    }

    pub fn is_keyword(&self, kw: Keyword) -> bool {
        self.kind == TokenKind::Keyword(kw)
    }
}

/// All token types
#[derive(Debug, Clone)]
pub enum TokenKind {
    // ==== Literals ====
    /// Integer literal: 42, 0xFF, 0b101, 1_000_000
    Int(i128),
    /// Float literal: 3.14, 1e10, 2.5e-3
    Float(f64),
    /// String literal: "hello"
    Str(String),
    /// Raw string: r#"hello"#
    RawStr(String),
    /// Byte string: b"hello"
    ByteStr(Vec<u8>),
    /// Character literal: 'a', '\n'
    Char(char),
    /// Byte literal: b'a'
    Byte(u8),

    // ==== Identifiers and Keywords ====
    /// Identifier: foo, _bar, Type123
    Ident(String),
    /// Keyword
    Keyword(Keyword),
    /// Lifetime: 'a, 'static
    Lifetime(String),

    // ==== Delimiters ====
    /// (
    LParen,
    /// )
    RParen,
    /// {
    LBrace,
    /// }
    RBrace,
    /// [
    LBracket,
    /// ]
    RBracket,

    // ==== Punctuation ====
    /// ,
    Comma,
    /// ;
    Semi,
    /// :
    Colon,
    /// ::
    PathSep,
    /// ->
    Arrow,
    /// =>
    FatArrow,
    /// .
    Dot,
    /// ..
    DotDot,
    /// ..=
    DotDotEq,
    /// ...
    Ellipsis,
    /// @
    At,
    /// #
    Pound,
    /// $
    Dollar,
    /// ?
    Question,
    /// _
    Underscore,

    // ==== Operators ====
    /// =
    Eq,
    /// ==
    EqEq,
    /// !=
    Ne,
    /// <
    Lt,
    /// <=
    Le,
    /// >
    Gt,
    /// >=
    Ge,
    /// +
    Plus,
    /// -
    Minus,
    /// *
    Star,
    /// /
    Slash,
    /// %
    Percent,
    /// &
    And,
    /// &&
    AndAnd,
    /// |
    Or,
    /// ||
    OrOr,
    /// ^
    Caret,
    /// !
    Not,
    /// ~
    Tilde,
    /// <<
    Shl,
    /// >>
    Shr,

    // ==== Compound Assignment ====
    /// +=
    PlusEq,
    /// -=
    MinusEq,
    /// *=
    StarEq,
    /// /=
    SlashEq,
    /// %=
    PercentEq,
    /// &=
    AndEq,
    /// |=
    OrEq,
    /// ^=
    CaretEq,
    /// <<=
    ShlEq,
    /// >>=
    ShrEq,

    // ==== Special ====
    /// End of file
    Eof,
    /// Unknown/invalid character
    Unknown(char),
}

impl TokenKind {
    /// Check if this is an identifier (not a keyword)
    pub fn is_ident(&self) -> bool {
        matches!(self, TokenKind::Ident(_))
    }

    /// Get identifier string if this is an identifier
    pub fn as_ident(&self) -> Option<&str> {
        match self {
            TokenKind::Ident(s) => Some(s),
            _ => None,
        }
    }

    /// Check if this token can start an expression
    pub fn can_start_expr(&self) -> bool {
        matches!(
            self,
            TokenKind::Int(_)
                | TokenKind::Float(_)
                | TokenKind::Str(_)
                | TokenKind::Char(_)
                | TokenKind::Ident(_)
                | TokenKind::LParen
                | TokenKind::LBracket
                | TokenKind::LBrace
                | TokenKind::Or      // closure |x|
                | TokenKind::OrOr    // closure ||
                | TokenKind::Not
                | TokenKind::Minus
                | TokenKind::Star    // deref
                | TokenKind::And     // borrow
                | TokenKind::AndAnd  // double borrow
                | TokenKind::Keyword(Keyword::If)
                | TokenKind::Keyword(Keyword::Match)
                | TokenKind::Keyword(Keyword::Loop)
                | TokenKind::Keyword(Keyword::While)
                | TokenKind::Keyword(Keyword::For)
                | TokenKind::Keyword(Keyword::Return)
                | TokenKind::Keyword(Keyword::Break)
                | TokenKind::Keyword(Keyword::Continue)
                | TokenKind::Keyword(Keyword::Move)
                | TokenKind::Keyword(Keyword::Unsafe)
                | TokenKind::Keyword(Keyword::Box)
        )
    }

    /// Check if this token can start a type
    pub fn can_start_type(&self) -> bool {
        matches!(
            self,
            TokenKind::Ident(_)
                | TokenKind::LParen      // tuple or unit
                | TokenKind::LBracket    // array/slice
                | TokenKind::Not         // never type !
                | TokenKind::Star        // raw pointer
                | TokenKind::And         // reference
                | TokenKind::Keyword(Keyword::Fn)
                | TokenKind::Keyword(Keyword::Impl)
                | TokenKind::Keyword(Keyword::Dyn)
                | TokenKind::Keyword(Keyword::SelfType)
        )
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Int(n) => write!(f, "{}", n),
            TokenKind::Float(n) => write!(f, "{}", n),
            TokenKind::Str(s) => write!(f, "\"{}\"", s),
            TokenKind::RawStr(s) => write!(f, "r\"{}\"", s),
            TokenKind::ByteStr(_) => write!(f, "b\"...\""),
            TokenKind::Char(c) => write!(f, "'{}'", c),
            TokenKind::Byte(b) => write!(f, "b'{}'", *b as char),
            TokenKind::Ident(s) => write!(f, "{}", s),
            TokenKind::Keyword(kw) => write!(f, "{}", kw.as_str()),
            TokenKind::Lifetime(s) => write!(f, "'{}", s),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Semi => write!(f, ";"),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::PathSep => write!(f, "::"),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::FatArrow => write!(f, "=>"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::DotDot => write!(f, ".."),
            TokenKind::DotDotEq => write!(f, "..="),
            TokenKind::Ellipsis => write!(f, "..."),
            TokenKind::At => write!(f, "@"),
            TokenKind::Pound => write!(f, "#"),
            TokenKind::Dollar => write!(f, "$"),
            TokenKind::Question => write!(f, "?"),
            TokenKind::Underscore => write!(f, "_"),
            TokenKind::Eq => write!(f, "="),
            TokenKind::EqEq => write!(f, "=="),
            TokenKind::Ne => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Le => write!(f, "<="),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::Ge => write!(f, ">="),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::And => write!(f, "&"),
            TokenKind::AndAnd => write!(f, "&&"),
            TokenKind::Or => write!(f, "|"),
            TokenKind::OrOr => write!(f, "||"),
            TokenKind::Caret => write!(f, "^"),
            TokenKind::Not => write!(f, "!"),
            TokenKind::Tilde => write!(f, "~"),
            TokenKind::Shl => write!(f, "<<"),
            TokenKind::Shr => write!(f, ">>"),
            TokenKind::PlusEq => write!(f, "+="),
            TokenKind::MinusEq => write!(f, "-="),
            TokenKind::StarEq => write!(f, "*="),
            TokenKind::SlashEq => write!(f, "/="),
            TokenKind::PercentEq => write!(f, "%="),
            TokenKind::AndEq => write!(f, "&="),
            TokenKind::OrEq => write!(f, "|="),
            TokenKind::CaretEq => write!(f, "^="),
            TokenKind::ShlEq => write!(f, "<<="),
            TokenKind::ShrEq => write!(f, ">>="),
            TokenKind::Eof => write!(f, "<EOF>"),
            TokenKind::Unknown(c) => write!(f, "<unknown '{}'>", c),
        }
    }
}

// Manual PartialEq implementation because f64 doesn't derive PartialEq in the way we need
impl PartialEq for TokenKind {
    fn eq(&self, other: &Self) -> bool {
        use TokenKind::*;
        match (self, other) {
            (Int(a), Int(b)) => a == b,
            (Float(a), Float(b)) => a.to_bits() == b.to_bits(),
            (Str(a), Str(b)) => a == b,
            (RawStr(a), RawStr(b)) => a == b,
            (ByteStr(a), ByteStr(b)) => a == b,
            (Char(a), Char(b)) => a == b,
            (Byte(a), Byte(b)) => a == b,
            (Ident(a), Ident(b)) => a == b,
            (Keyword(a), Keyword(b)) => a == b,
            (Lifetime(a), Lifetime(b)) => a == b,
            (LParen, LParen) => true,
            (RParen, RParen) => true,
            (LBrace, LBrace) => true,
            (RBrace, RBrace) => true,
            (LBracket, LBracket) => true,
            (RBracket, RBracket) => true,
            (Comma, Comma) => true,
            (Semi, Semi) => true,
            (Colon, Colon) => true,
            (PathSep, PathSep) => true,
            (Arrow, Arrow) => true,
            (FatArrow, FatArrow) => true,
            (Dot, Dot) => true,
            (DotDot, DotDot) => true,
            (DotDotEq, DotDotEq) => true,
            (Ellipsis, Ellipsis) => true,
            (At, At) => true,
            (Pound, Pound) => true,
            (Dollar, Dollar) => true,
            (Question, Question) => true,
            (Underscore, Underscore) => true,
            (Eq, Eq) => true,
            (EqEq, EqEq) => true,
            (Ne, Ne) => true,
            (Lt, Lt) => true,
            (Le, Le) => true,
            (Gt, Gt) => true,
            (Ge, Ge) => true,
            (Plus, Plus) => true,
            (Minus, Minus) => true,
            (Star, Star) => true,
            (Slash, Slash) => true,
            (Percent, Percent) => true,
            (And, And) => true,
            (AndAnd, AndAnd) => true,
            (Or, Or) => true,
            (OrOr, OrOr) => true,
            (Caret, Caret) => true,
            (Not, Not) => true,
            (Tilde, Tilde) => true,
            (Shl, Shl) => true,
            (Shr, Shr) => true,
            (PlusEq, PlusEq) => true,
            (MinusEq, MinusEq) => true,
            (StarEq, StarEq) => true,
            (SlashEq, SlashEq) => true,
            (PercentEq, PercentEq) => true,
            (AndEq, AndEq) => true,
            (OrEq, OrEq) => true,
            (CaretEq, CaretEq) => true,
            (ShlEq, ShlEq) => true,
            (ShrEq, ShrEq) => true,
            (Eof, Eof) => true,
            (Unknown(a), Unknown(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for TokenKind {}

impl Hash for TokenKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the discriminant first
        std::mem::discriminant(self).hash(state);
        // Then hash the payload for variants that have one
        use TokenKind::*;
        match self {
            Int(n) => n.hash(state),
            Float(f) => f.to_bits().hash(state),
            Str(s) | RawStr(s) | Ident(s) | Lifetime(s) => s.hash(state),
            ByteStr(b) => b.hash(state),
            Char(c) => c.hash(state),
            Byte(b) => b.hash(state),
            Keyword(k) => k.hash(state),
            Unknown(c) => c.hash(state),
            // Unit variants - discriminant is enough
            _ => {}
        }
    }
}

/// All Rust/Bolt keywords
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    // Item keywords
    As,
    Async,
    Await,
    Box,
    Break,
    Const,
    Continue,
    Crate,
    Dyn,
    Else,
    Enum,
    Extern,
    False,
    Fn,
    For,
    If,
    Impl,
    In,
    Let,
    Loop,
    Match,
    Mod,
    Move,
    Mut,
    Pub,
    Ref,
    Return,
    SelfLower,  // self
    SelfType,   // Self
    Static,
    Struct,
    Super,
    Trait,
    True,
    Type,
    Unsafe,
    Use,
    Where,
    While,
}

impl Keyword {
    /// Get the string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Keyword::As => "as",
            Keyword::Async => "async",
            Keyword::Await => "await",
            Keyword::Box => "box",
            Keyword::Break => "break",
            Keyword::Const => "const",
            Keyword::Continue => "continue",
            Keyword::Crate => "crate",
            Keyword::Dyn => "dyn",
            Keyword::Else => "else",
            Keyword::Enum => "enum",
            Keyword::Extern => "extern",
            Keyword::False => "false",
            Keyword::Fn => "fn",
            Keyword::For => "for",
            Keyword::If => "if",
            Keyword::Impl => "impl",
            Keyword::In => "in",
            Keyword::Let => "let",
            Keyword::Loop => "loop",
            Keyword::Match => "match",
            Keyword::Mod => "mod",
            Keyword::Move => "move",
            Keyword::Mut => "mut",
            Keyword::Pub => "pub",
            Keyword::Ref => "ref",
            Keyword::Return => "return",
            Keyword::SelfLower => "self",
            Keyword::SelfType => "Self",
            Keyword::Static => "static",
            Keyword::Struct => "struct",
            Keyword::Super => "super",
            Keyword::Trait => "trait",
            Keyword::True => "true",
            Keyword::Type => "type",
            Keyword::Unsafe => "unsafe",
            Keyword::Use => "use",
            Keyword::Where => "where",
            Keyword::While => "while",
        }
    }

    /// Try to parse a keyword from a string
    pub fn from_str(s: &str) -> Option<Keyword> {
        match s {
            "as" => Some(Keyword::As),
            "async" => Some(Keyword::Async),
            "await" => Some(Keyword::Await),
            "box" => Some(Keyword::Box),
            "break" => Some(Keyword::Break),
            "const" => Some(Keyword::Const),
            "continue" => Some(Keyword::Continue),
            "crate" => Some(Keyword::Crate),
            "dyn" => Some(Keyword::Dyn),
            "else" => Some(Keyword::Else),
            "enum" => Some(Keyword::Enum),
            "extern" => Some(Keyword::Extern),
            "false" => Some(Keyword::False),
            "fn" => Some(Keyword::Fn),
            "for" => Some(Keyword::For),
            "if" => Some(Keyword::If),
            "impl" => Some(Keyword::Impl),
            "in" => Some(Keyword::In),
            "let" => Some(Keyword::Let),
            "loop" => Some(Keyword::Loop),
            "match" => Some(Keyword::Match),
            "mod" => Some(Keyword::Mod),
            "move" => Some(Keyword::Move),
            "mut" => Some(Keyword::Mut),
            "pub" => Some(Keyword::Pub),
            "ref" => Some(Keyword::Ref),
            "return" => Some(Keyword::Return),
            "self" => Some(Keyword::SelfLower),
            "Self" => Some(Keyword::SelfType),
            "static" => Some(Keyword::Static),
            "struct" => Some(Keyword::Struct),
            "super" => Some(Keyword::Super),
            "trait" => Some(Keyword::Trait),
            "true" => Some(Keyword::True),
            "type" => Some(Keyword::Type),
            "unsafe" => Some(Keyword::Unsafe),
            "use" => Some(Keyword::Use),
            "where" => Some(Keyword::Where),
            "while" => Some(Keyword::While),
            _ => None,
        }
    }
}
