//! Bolt Parser - Built with Chumsky
//!
//! # Attribution
//! This parser uses [Chumsky](https://github.com/zesterer/chumsky) by Joshua Barretto (zesterer).
//! Chumsky is an excellent parser combinator library with first-class error recovery.
//! Thank you to the Chumsky maintainers and contributors!
//!
//! # Architecture
//! ```text
//! Source -> Lexer (rustc_lexer) -> Tokens -> Parser (chumsky) -> AST -> HIR
//! ```

pub mod ast;
pub mod lower;

use crate::lexer::{cook_tokens, Lexer, Span, Token, TokenKind, Keyword};
use ast::*;
use chumsky::prelude::*;

/// Parse error with span
pub type ParseError = Simple<TokenKind>;

/// Result type for parsing
pub type ParseResult<T> = Result<T, Vec<ParseError>>;

/// Parse source code into AST
pub fn parse(source: &str) -> (Option<SourceFile>, Vec<ParseError>) {
    let (tokens, lex_errors) = Lexer::tokenize(source);
    let tokens = cook_tokens(tokens);

    // Convert lex errors to parse errors
    let mut errors: Vec<ParseError> = lex_errors
        .iter()
        .map(|e| Simple::custom(e.span.start as usize..e.span.end as usize, &e.message))
        .collect();

    // Create token stream for chumsky
    // Filter out EOF tokens - Chumsky expects stream to end, not an EOF token
    let len = source.len();
    let token_stream: Vec<(TokenKind, std::ops::Range<usize>)> = tokens
        .into_iter()
        .filter(|t| !matches!(t.kind, TokenKind::Eof))
        .map(|t| (t.kind, t.span.start as usize..t.span.end as usize))
        .collect();

    let (ast, parse_errors) = source_file_parser()
        .parse_recovery(chumsky::Stream::from_iter(
            len..len + 1,
            token_stream.into_iter(),
        ));

    errors.extend(parse_errors);
    (ast, errors)
}

/// Helper to create span from range
fn span_from_range(range: std::ops::Range<usize>) -> Span {
    Span::new(range.start as u32, range.end as u32)
}

// ============================================================================
// Token Matchers
// ============================================================================

fn ident() -> impl Parser<TokenKind, Ident, Error = ParseError> + Clone {
    select! {
        TokenKind::Ident(name) => name,
    }
    .map_with_span(|name, span| Ident::new(name, span_from_range(span)))
    .labelled("identifier")
}

fn keyword(kw: Keyword) -> impl Parser<TokenKind, (), Error = ParseError> + Clone {
    just(TokenKind::Keyword(kw)).ignored()
}

fn lit() -> impl Parser<TokenKind, Lit, Error = ParseError> + Clone {
    select! {
        TokenKind::Int(n) => Lit::Int(n, None),
        TokenKind::Float(n) => Lit::Float(n, None),
        TokenKind::Str(s) => Lit::Str(s),
        TokenKind::Char(c) => Lit::Char(c),
        TokenKind::Byte(b) => Lit::Byte(b),
        TokenKind::ByteStr(b) => Lit::ByteStr(b),
        TokenKind::Keyword(Keyword::True) => Lit::Bool(true),
        TokenKind::Keyword(Keyword::False) => Lit::Bool(false),
    }
    .labelled("literal")
}

fn lifetime() -> impl Parser<TokenKind, Ident, Error = ParseError> + Clone {
    select! {
        TokenKind::Lifetime(name) => name,
    }
    .map_with_span(|name, span| Ident::new(name, span_from_range(span)))
    .labelled("lifetime")
}

// ============================================================================
// Type Parser
// ============================================================================

fn type_parser() -> impl Parser<TokenKind, Type, Error = ParseError> + Clone {
    recursive(|ty| {
        // Path type with generics - uses the recursive ty handle for generic args
        let generic_args = ty.clone()
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .delimited_by(just(TokenKind::Lt), just(TokenKind::Gt))
            .map(|args| GenericArgs {
                args: args.into_iter().map(GenericArg::Type).collect(),
            });

        // Type segment can be an identifier or Self keyword
        let type_ident = choice((
            keyword(Keyword::SelfType).map_with_span(|_, span| Ident::new("Self".to_string(), span_from_range(span))),
            ident(),
        ));

        let segment = type_ident
            .then(generic_args.or_not())
            .map(|(ident, args)| PathSegment { ident, args });

        let path_type = segment
            .separated_by(just(TokenKind::PathSep))
            .at_least(1)
            .map_with_span(|segments, span| Path {
                segments,
                span: span_from_range(span),
            })
            .map(TypeKind::Path);

        let ref_type = just(TokenKind::And)
            .ignore_then(lifetime().or_not())
            .then(keyword(Keyword::Mut).or_not().map(|m| m.is_some()))
            .then(ty.clone())
            .map(|((lifetime, mutable), inner)| TypeKind::Ref {
                lifetime,
                mutable,
                inner: Box::new(inner),
            });

        let ptr_type = just(TokenKind::Star)
            .ignore_then(
                keyword(Keyword::Const).to(false)
                    .or(keyword(Keyword::Mut).to(true))
            )
            .then(ty.clone())
            .map(|(mutable, inner)| TypeKind::Ptr {
                mutable,
                inner: Box::new(inner),
            });

        let slice_type = ty.clone()
            .delimited_by(just(TokenKind::LBracket), just(TokenKind::RBracket))
            .map(|inner| TypeKind::Slice(Box::new(inner)));

        let tuple_type = ty.clone()
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
            .map(TypeKind::Tuple);

        let never_type = just(TokenKind::Not)
            .map(|_| TypeKind::Never);

        let infer_type = just(TokenKind::Underscore)
            .map(|_| TypeKind::Infer);

        let impl_trait = keyword(Keyword::Impl)
            .ignore_then(type_bounds_with(ty.clone()))
            .map(TypeKind::ImplTrait);

        let dyn_trait = keyword(Keyword::Dyn)
            .ignore_then(type_bounds_with(ty.clone()))
            .map(TypeKind::DynTrait);

        // Function pointer type: fn(Type1, Type2) -> RetType
        let fn_ptr_type = keyword(Keyword::Fn)
            .ignore_then(
                ty.clone()
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
            )
            .then(
                just(TokenKind::Arrow)
                    .ignore_then(ty.clone())
                    .or_not()
            )
            .map(|(params, ret)| TypeKind::Fn {
                params,
                ret: Box::new(ret.unwrap_or_else(|| Type {
                    kind: TypeKind::Tuple(vec![]),
                    span: Span::dummy(),
                })),
            });

        choice((
            fn_ptr_type,
            ref_type,
            ptr_type,
            impl_trait,
            dyn_trait,
            never_type,
            infer_type,
            tuple_type,
            slice_type,
            path_type,
        ))
        .map_with_span(|kind, span| Type { kind, span: span_from_range(span) })
    })
}

fn type_bounds() -> impl Parser<TokenKind, Vec<TypeBound>, Error = ParseError> + Clone {
    let trait_bound = path_parser().map(TypeBound::Trait);
    let lifetime_bound = lifetime().map(TypeBound::Lifetime);

    choice((trait_bound, lifetime_bound))
        .separated_by(just(TokenKind::Plus))
        .at_least(1)
}

/// Type bounds with generic args support (for impl Trait, dyn Trait)
fn type_bounds_with(
    ty: impl Parser<TokenKind, Type, Error = ParseError> + Clone,
) -> impl Parser<TokenKind, Vec<TypeBound>, Error = ParseError> + Clone {
    // Generic args parser
    let generic_args = ty
        .separated_by(just(TokenKind::Comma))
        .allow_trailing()
        .delimited_by(just(TokenKind::Lt), just(TokenKind::Gt))
        .map(|args| GenericArgs {
            args: args.into_iter().map(GenericArg::Type).collect(),
        });

    // Path segment with optional generic args
    let type_ident = choice((
        keyword(Keyword::SelfType).map_with_span(|_, span| Ident::new("Self".to_string(), span_from_range(span))),
        ident(),
    ));

    let segment = type_ident
        .then(generic_args.or_not())
        .map(|(ident, args)| PathSegment { ident, args });

    // Full path with generics
    let path_with_generics = segment
        .separated_by(just(TokenKind::PathSep))
        .at_least(1)
        .map_with_span(|segments, span| Path {
            segments,
            span: span_from_range(span),
        });

    let trait_bound = path_with_generics.map(TypeBound::Trait);
    let lifetime_bound = lifetime().map(TypeBound::Lifetime);

    choice((trait_bound, lifetime_bound))
        .separated_by(just(TokenKind::Plus))
        .at_least(1)
}

// ============================================================================
// Path Parser
// ============================================================================

fn path_parser() -> impl Parser<TokenKind, Path, Error = ParseError> + Clone {
    // Simple path parser for expressions - no generic args to avoid mutual recursion
    // Turbofish syntax (e.g., Vec::<i32>::new()) can be handled separately if needed

    // Path segment can be an identifier or a keyword like crate/self/super/Self
    let path_segment = choice((
        // Keywords that can appear in paths
        keyword(Keyword::Crate).map_with_span(|_, span| Ident::new("crate".to_string(), span_from_range(span))),
        keyword(Keyword::SelfLower).map_with_span(|_, span| Ident::new("self".to_string(), span_from_range(span))),
        keyword(Keyword::Super).map_with_span(|_, span| Ident::new("super".to_string(), span_from_range(span))),
        keyword(Keyword::SelfType).map_with_span(|_, span| Ident::new("Self".to_string(), span_from_range(span))),
        // Regular identifier
        ident(),
    )).map(|ident| PathSegment { ident, args: None });

    path_segment
        .separated_by(just(TokenKind::PathSep))
        .at_least(1)
        .map_with_span(|segments, span| Path {
            segments,
            span: span_from_range(span),
        })
}

// ============================================================================
// Pattern Parser
// ============================================================================

// Simple pattern parser for closure params (no or-patterns to avoid conflict with |)
// Uses recursion to support nested tuple patterns like (a, (b, c))
fn simple_pattern_parser() -> impl Parser<TokenKind, Pattern, Error = ParseError> + Clone {
    recursive(|pat| {
        let wild = just(TokenKind::Underscore)
            .map(|_| PatternKind::Wild);

        let lit_pat = lit().map(PatternKind::Lit);

        let ident_pat = keyword(Keyword::Ref).or_not()
            .then(keyword(Keyword::Mut).or_not())
            .then(ident())
            .map(|((by_ref, mutable), name)| PatternKind::Ident {
                by_ref: by_ref.is_some(),
                mutable: mutable.is_some(),
                name,
                subpat: None,
            });

        let ref_pat = just(TokenKind::And)
            .ignore_then(keyword(Keyword::Mut).or_not().map(|m| m.is_some()))
            .then(pat.clone())
            .map(|(mutable, inner)| PatternKind::Ref {
                mutable,
                inner: Box::new(inner),
            });

        // Tuple pattern with full nesting: (a, (b, c), _)
        let tuple_pat = pat.clone()
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
            .map(PatternKind::Tuple);

        choice((
            wild,
            ref_pat,
            tuple_pat,
            lit_pat,
            ident_pat,
        ))
        .map_with_span(|kind, span| Pattern { kind, span: span_from_range(span) })
    })
}

fn pattern_parser() -> impl Parser<TokenKind, Pattern, Error = ParseError> + Clone {
    recursive(|pat| {
        let wild = just(TokenKind::Underscore)
            .map(|_| PatternKind::Wild);

        let lit_pat = lit().map(PatternKind::Lit);

        let ident_pat = keyword(Keyword::Ref).or_not()
            .then(keyword(Keyword::Mut).or_not())
            .then(ident())
            .then(just(TokenKind::At).ignore_then(pat.clone()).or_not())
            .map(|(((by_ref, mutable), name), subpat)| PatternKind::Ident {
                by_ref: by_ref.is_some(),
                mutable: mutable.is_some(),
                name,
                subpat: subpat.map(Box::new),
            });

        let tuple_pat = pat.clone()
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
            .map(PatternKind::Tuple);

        let ref_pat = just(TokenKind::And)
            .ignore_then(keyword(Keyword::Mut).or_not().map(|m| m.is_some()))
            .then(pat.clone())
            .map(|(mutable, inner)| PatternKind::Ref {
                mutable,
                inner: Box::new(inner),
            });

        // Tuple struct pattern: Path(pat, pat, ...)
        let tuple_struct_pat = path_parser()
            .then(
                pat.clone()
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
            )
            .map(|(path, fields)| PatternKind::TupleStruct { path, fields });

        // Struct pattern: Path { field, field: pat, .. }
        let field_pat = ident()
            .then(just(TokenKind::Colon).ignore_then(pat.clone()).or_not())
            .map_with_span(|(name, pat), span| {
                let shorthand = pat.is_none();
                let pattern = pat.unwrap_or_else(|| Pattern {
                    kind: PatternKind::Ident {
                        by_ref: false,
                        mutable: false,
                        name: name.clone(),
                        subpat: None,
                    },
                    span: span_from_range(span),
                });
                FieldPat {
                    name,
                    pattern,
                    shorthand,
                }
            });
        let struct_pat = path_parser()
            .then(
                // Fields with optional rest pattern (..)
                // Handle: { }, { x }, { x, }, { x, .. }, { .. }
                field_pat
                    .separated_by(just(TokenKind::Comma))
                    .then(
                        just(TokenKind::Comma).or_not()
                            .ignore_then(just(TokenKind::DotDot))
                            .or_not()
                    )
                    .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
            )
            .map(|(path, (fields, rest))| PatternKind::Struct {
                path,
                fields,
                rest: rest.is_some(),
            });

        let path_pat = path_parser().map(PatternKind::Path);

        // Qualified path pattern: Path::Variant (without parens/braces)
        // Must check for :: to distinguish from simple ident bindings
        let qualified_path_pat = path_parser()
            .try_map(|path, span| {
                if path.segments.len() > 1 {
                    Ok(PatternKind::Path(path))
                } else {
                    Err(Simple::custom(span, "not a qualified path"))
                }
            });

        let base = choice((
            wild,
            ref_pat,
            tuple_pat,
            lit_pat,
            tuple_struct_pat,  // Must come before ident_pat to handle Path(...)
            struct_pat,        // Must come before ident_pat to handle Path{...}
            qualified_path_pat, // Qualified paths like Option::None before ident
            ident_pat,
            path_pat,          // Single-segment paths like None
        ));

        // Or patterns: a | b | c
        base.clone()
            .separated_by(just(TokenKind::Or))
            .at_least(1)
            .map(|pats| {
                if pats.len() == 1 {
                    pats.into_iter().next().unwrap()
                } else {
                    PatternKind::Or(pats.into_iter().map(|kind| Pattern {
                        kind,
                        span: Span::dummy(),
                    }).collect())
                }
            })
            .map_with_span(|kind, span| Pattern { kind, span: span_from_range(span) })
    })
}

// ============================================================================
// Balanced Delimiter Parser (for macros)
// ============================================================================

/// Parser for balanced delimiters - handles nested braces/parens/brackets
/// Defined separately to avoid nested recursive parser construction
fn balanced_tokens_parser() -> impl Parser<TokenKind, (), Error = ParseError> + Clone {
    recursive::<_, (), _, _, _>(|balanced| {
        let nested_brace = just(TokenKind::LBrace)
            .ignore_then(balanced.clone().repeated())
            .then_ignore(just(TokenKind::RBrace))
            .map(|_| ());
        let nested_paren = just(TokenKind::LParen)
            .ignore_then(balanced.clone().repeated())
            .then_ignore(just(TokenKind::RParen))
            .map(|_| ());
        let nested_bracket = just(TokenKind::LBracket)
            .ignore_then(balanced.clone().repeated())
            .then_ignore(just(TokenKind::RBracket))
            .map(|_| ());
        let other = none_of([
            TokenKind::LBrace, TokenKind::RBrace,
            TokenKind::LParen, TokenKind::RParen,
            TokenKind::LBracket, TokenKind::RBracket,
        ]).ignored();
        choice((nested_brace, nested_paren, nested_bracket, other))
    })
}

// ============================================================================
// Expression Parser
// ============================================================================

fn expr_parser_with(
    ty: impl Parser<TokenKind, Type, Error = ParseError> + Clone + 'static,
    pat: impl Parser<TokenKind, Pattern, Error = ParseError> + Clone + 'static,
    simple_pat: impl Parser<TokenKind, Pattern, Error = ParseError> + Clone + 'static,
    balanced_tokens: impl Parser<TokenKind, (), Error = ParseError> + Clone + 'static,
) -> impl Parser<TokenKind, Expr, Error = ParseError> + Clone {
    recursive(move |expr| {
        let balanced_tokens = balanced_tokens.clone();
        let ty = ty.clone();
        let pat = pat.clone();
        let simple_pat = simple_pat.clone();
        let block = block_parser(expr.clone(), ty.clone(), pat.clone());

        // Atoms
        let lit_expr = lit().map(ExprKind::Lit);

        // Macro call: path!(args) or path![args] or path!{args}
        // Excludes macro_rules! which is a definition, not an invocation
        let macro_call = path_parser()
            .try_map(|path, span| {
                // Reject macro_rules! as it's a definition, not invocation
                if path.segments.len() == 1 && path.segments[0].ident.name == "macro_rules" {
                    Err(Simple::custom(span, "macro_rules is a definition"))
                } else {
                    Ok(path)
                }
            })
            .then_ignore(just(TokenKind::Not))
            .then(
                just(TokenKind::LParen)
                    .ignore_then(balanced_tokens.clone().repeated())
                    .then_ignore(just(TokenKind::RParen))
                    .or(just(TokenKind::LBracket)
                        .ignore_then(balanced_tokens.clone().repeated())
                        .then_ignore(just(TokenKind::RBracket)))
                    .or(just(TokenKind::LBrace)
                        .ignore_then(balanced_tokens.clone().repeated())
                        .then_ignore(just(TokenKind::RBrace)))
            )
            .map(|(path, _)| ExprKind::MacroCall { path, args: String::new() });

        // macro_rules! name { ... } as an expression (skipped)
        let macro_rules_expr = just(TokenKind::Ident("macro_rules".to_string()))
            .ignore_then(just(TokenKind::Not))
            .ignore_then(ident())
            .then(
                just(TokenKind::LBrace)
                    .ignore_then(balanced_tokens.clone().repeated())
                    .then_ignore(just(TokenKind::RBrace))
                    .or(just(TokenKind::LParen)
                        .ignore_then(balanced_tokens.clone().repeated())
                        .then_ignore(just(TokenKind::RParen)))
            )
            .map_with_span(|(name, _), span| ExprKind::MacroCall {
                path: Path {
                    segments: vec![PathSegment {
                        ident: Ident::new("macro_rules".to_string(), span_from_range(span.clone())),
                        args: None,
                    }],
                    span: span_from_range(span),
                },
                args: String::new(),
            });

        let path_expr = path_parser().map(ExprKind::Path);

        // self as an expression (for method bodies)
        let self_expr = keyword(Keyword::SelfLower).map_with_span(|_, span| {
            let s = span_from_range(span);
            ExprKind::Path(Path {
                segments: vec![PathSegment {
                    ident: Ident::new("self".to_string(), s),
                    args: None,
                }],
                span: s,
            })
        });

        let tuple_expr = expr.clone()
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
            .map(|exprs| {
                if exprs.len() == 1 {
                    exprs.into_iter().next().unwrap().kind
                } else {
                    ExprKind::Tuple(exprs)
                }
            });

        let array_expr = expr.clone()
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .delimited_by(just(TokenKind::LBracket), just(TokenKind::RBracket))
            .map(|exprs| ExprKind::Array(ArrayExpr::List(exprs)));

        // ========================================================================
        // Condition expression parser (for if/while/for conditions)
        // This is built BEFORE if_expr to break the circular dependency.
        // It excludes struct literals at the top level to avoid ambiguity.
        // ========================================================================

        // Atoms usable in conditions (before control flow is defined)
        let cond_atom = choice((
            lit_expr.clone(),
            self_expr.clone(),
            macro_rules_expr.clone(),
            macro_call.clone(),
            path_expr.clone(),
            tuple_expr.clone(),  // uses expr.clone() for nested - that's fine
            array_expr.clone(),
        ))
        .map_with_span(|kind, span| Expr { kind, span: span_from_range(span) })
        .boxed();

        // Turbofish for conditions: ::<Type1, Type2>
        let cond_turbofish = just(TokenKind::PathSep)
            .ignore_then(
                ty.clone()
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::Lt), just(TokenKind::Gt))
            )
            .map(|args| GenericArgs {
                args: args.into_iter().map(GenericArg::Type).collect(),
            });

        // Postfix operations for conditions (without struct literals)
        #[derive(Clone)]
        enum CondPostfixOp {
            Field(Ident),
            MethodCall(Ident, Option<GenericArgs>, Vec<Expr>),
            Call(Vec<Expr>),
            Index(Expr),
            Try,
        }

        let cond_postfix_op = choice((
            // Method call or field access
            just(TokenKind::Dot)
                .ignore_then(
                    ident()
                        .or(select! { TokenKind::Int(n) => n }
                            .map_with_span(|n, span| Ident::new(n.to_string(), span_from_range(span))))
                )
                .then(cond_turbofish.clone().or_not())
                .then(
                    expr.clone()
                        .separated_by(just(TokenKind::Comma))
                        .allow_trailing()
                        .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
                        .or_not()
                )
                .map(|((field, turbo), args)| {
                    match args {
                        Some(args) => CondPostfixOp::MethodCall(field, turbo, args),
                        None if turbo.is_some() => CondPostfixOp::MethodCall(field, turbo, vec![]),
                        None => CondPostfixOp::Field(field),
                    }
                }),
            // Call: (args)
            expr.clone()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
                .map(CondPostfixOp::Call),
            // Index: [expr]
            expr.clone()
                .delimited_by(just(TokenKind::LBracket), just(TokenKind::RBracket))
                .map(CondPostfixOp::Index),
            // Try: ?
            just(TokenKind::Question).to(CondPostfixOp::Try),
        ));

        let cond_postfix = cond_atom.clone()
            .then(cond_postfix_op.repeated())
            .foldl(|e: Expr, op: CondPostfixOp| -> Expr {
                let span = e.span;
                let kind = match op {
                    CondPostfixOp::Field(field) => ExprKind::Field {
                        expr: Box::new(e),
                        field,
                    },
                    CondPostfixOp::MethodCall(method, turbofish, args) => ExprKind::MethodCall {
                        receiver: Box::new(e),
                        method,
                        turbofish,
                        args,
                    },
                    CondPostfixOp::Call(args) => ExprKind::Call {
                        func: Box::new(e),
                        args,
                    },
                    CondPostfixOp::Index(index) => ExprKind::Index {
                        expr: Box::new(e),
                        index: Box::new(index),
                    },
                    CondPostfixOp::Try => ExprKind::Try(Box::new(e)),
                };
                Expr { kind, span }
            });

        // Unary prefix for conditions
        let cond_prefix_op = choice((
            just(TokenKind::Minus).to(UnaryOp::Neg),
            just(TokenKind::Not).to(UnaryOp::Not),
            just(TokenKind::Star).to(UnaryOp::Deref),
        ));

        let cond_ref_op = just(TokenKind::And)
            .ignore_then(keyword(Keyword::Mut).or_not().map(|m| m.is_some()));

        // Build unary with foldr for right-associativity
        let cond_unary = cond_prefix_op.clone()
            .map(|op| (op, false))
            .or(cond_ref_op.map(|mutable| (if mutable { UnaryOp::RefMut } else { UnaryOp::Ref }, true)))
            .repeated()
            .then(cond_postfix.clone())
            .foldr(|(op, is_ref), expr: Expr| {
                let span = expr.span;
                let kind = if is_ref {
                    ExprKind::Ref { mutable: op == UnaryOp::RefMut, expr: Box::new(expr) }
                } else {
                    ExprKind::Unary { op, expr: Box::new(expr) }
                };
                Expr { span, kind }
            });

        // Type cast for conditions
        let cond_cast = cond_unary.clone()
            .then(keyword(Keyword::As).ignore_then(ty.clone()).repeated())
            .foldl(|expr, ty| Expr {
                span: expr.span,
                kind: ExprKind::Cast { expr: Box::new(expr), ty },
            });

        // Binary operators for conditions
        let cond_product = cond_cast.clone()
            .then(
                choice((
                    just(TokenKind::Star).to(BinOp::Mul),
                    just(TokenKind::Slash).to(BinOp::Div),
                    just(TokenKind::Percent).to(BinOp::Rem),
                ))
                .then(cond_cast.clone())
                .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let cond_sum = cond_product.clone()
            .then(
                choice((
                    just(TokenKind::Plus).to(BinOp::Add),
                    just(TokenKind::Minus).to(BinOp::Sub),
                ))
                .then(cond_product.clone())
                .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let cond_shift = cond_sum.clone()
            .then(
                choice((
                    just(TokenKind::Shl).to(BinOp::Shl),
                    just(TokenKind::Shr).to(BinOp::Shr),
                ))
                .then(cond_sum.clone())
                .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let cond_bit_and = cond_shift.clone()
            .then(
                just(TokenKind::And).to(BinOp::BitAnd)
                    .then(cond_shift.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let cond_bit_xor = cond_bit_and.clone()
            .then(
                just(TokenKind::Caret).to(BinOp::BitXor)
                    .then(cond_bit_and.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let cond_bit_or = cond_bit_xor.clone()
            .then(
                just(TokenKind::Or).to(BinOp::BitOr)
                    .then(cond_bit_xor.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let cond_comparison = cond_bit_or.clone()
            .then(
                choice((
                    just(TokenKind::EqEq).to(BinOp::Eq),
                    just(TokenKind::Ne).to(BinOp::Ne),
                    just(TokenKind::Lt).to(BinOp::Lt),
                    just(TokenKind::Le).to(BinOp::Le),
                    just(TokenKind::Gt).to(BinOp::Gt),
                    just(TokenKind::Ge).to(BinOp::Ge),
                ))
                .then(cond_bit_or.clone())
                .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let cond_logical_and = cond_comparison.clone()
            .then(
                just(TokenKind::AndAnd).to(BinOp::And)
                    .then(cond_comparison.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let cond_expr = cond_logical_and.clone()
            .then(
                just(TokenKind::OrOr).to(BinOp::Or)
                    .then(cond_logical_and.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        // ========================================================================
        // End of condition expression parser
        // ========================================================================

        // Let condition for if let / while let: let Pattern = expr
        let let_cond = keyword(Keyword::Let)
            .ignore_then(pat.clone())
            .then_ignore(just(TokenKind::Eq))
            .then(expr.clone())
            .map_with_span(|(pat, e), span| Expr {
                kind: ExprKind::Let { pat, expr: Box::new(e) },
                span: span_from_range(span),
            });

        let if_expr = keyword(Keyword::If)
            .ignore_then(let_cond.clone().or(cond_expr.clone()))
            .then(block.clone())
            .then(keyword(Keyword::Else).ignore_then(
                block.clone().map(|b| Expr { kind: ExprKind::Block(b), span: Span::dummy() })
                    .or(expr.clone())
            ).or_not())
            .map(|((cond, then_branch), else_branch)| ExprKind::If {
                cond: Box::new(cond),
                then_branch,
                else_branch: else_branch.map(Box::new),
            });

        let match_arm = pat.clone()
            .then(just(TokenKind::Keyword(Keyword::If)).ignore_then(cond_expr.clone()).or_not())
            .then_ignore(just(TokenKind::FatArrow))
            .then(expr.clone())
            .map_with_span(|((pattern, guard), body), span| MatchArm {
                pattern,
                guard: guard.map(Box::new),
                body,
                span: span_from_range(span),
            });

        let match_expr = keyword(Keyword::Match)
            .ignore_then(expr.clone())
            .then(
                // Match arms with optional commas (commas are optional after blocks)
                match_arm
                    .then_ignore(just(TokenKind::Comma).or_not())
                    .repeated()
                    .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
            )
            .map(|(scrutinee, arms)| ExprKind::Match {
                expr: Box::new(scrutinee),
                arms,
            });

        let loop_expr = keyword(Keyword::Loop)
            .ignore_then(block.clone())
            .map(|body| ExprKind::Loop { label: None, body });

        let while_expr = keyword(Keyword::While)
            .ignore_then(let_cond.clone().or(cond_expr.clone()))
            .then(block.clone())
            .map(|(cond, body)| ExprKind::While {
                label: None,
                cond: Box::new(cond),
                body,
            });

        let for_expr = keyword(Keyword::For)
            .ignore_then(pat.clone())
            .then_ignore(keyword(Keyword::In))
            .then(cond_expr.clone())
            .then(block.clone())
            .map(|((pat, iter), body)| ExprKind::For {
                label: None,
                pat,
                iter: Box::new(iter),
                body,
            });

        let return_expr = keyword(Keyword::Return)
            .ignore_then(expr.clone().or_not())
            .map(|e| ExprKind::Return(e.map(Box::new)));

        let break_expr = keyword(Keyword::Break)
            .ignore_then(expr.clone().or_not())
            .map(|e| ExprKind::Break { label: None, expr: e.map(Box::new) });

        let continue_expr = keyword(Keyword::Continue)
            .map(|_| ExprKind::Continue(None));

        // Closure: use simple_pat to avoid or-pattern conflict with | delimiter
        let closure = keyword(Keyword::Move).or_not()
            .then(
                simple_pat.clone()
                    .then(just(TokenKind::Colon).ignore_then(ty.clone()).or_not())
                    .map(|(p, t)| ClosureParam { pattern: p, ty: t })
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::Or), just(TokenKind::Or))
                    .or(just(TokenKind::OrOr).map(|_| Vec::new()))
            )
            .then(just(TokenKind::Arrow).ignore_then(ty.clone()).or_not())
            .then(expr.clone())
            .map(|(((is_move, params), ret_type), body)| ExprKind::Closure {
                is_move: is_move.is_some(),
                params,
                ret_type,
                body: Box::new(body),
            });

        let block_expr = block.clone().map(ExprKind::Block);

        // Unsafe block: unsafe { ... }
        let unsafe_expr = keyword(Keyword::Unsafe)
            .ignore_then(block.clone())
            .map(ExprKind::Unsafe);

        // Prefix range: ..expr, ..=expr, or just ..
        // This needs to use expr recursively, so we wrap in the right way
        let prefix_range = choice((
            just(TokenKind::DotDotEq)
                .ignore_then(expr.clone())
                .map(|end| ExprKind::Range {
                    start: None,
                    end: Some(Box::new(end)),
                    inclusive: true,
                }),
            just(TokenKind::DotDot)
                .ignore_then(expr.clone().or_not())
                .map(|end| ExprKind::Range {
                    start: None,
                    end: end.map(Box::new),
                    inclusive: false,
                }),
        ));

        // Primary expression
        let atom = choice((
            prefix_range,  // must come before other atoms
            lit_expr,
            if_expr,
            match_expr,
            loop_expr,
            while_expr,
            for_expr,
            return_expr,
            break_expr,
            continue_expr,
            closure,
            unsafe_expr,  // must come before block_expr
            block_expr,
            tuple_expr,
            array_expr,
            self_expr,  // must come before path_expr
            macro_rules_expr, // macro_rules! definitions (before macro_call)
            macro_call, // must come before path_expr
            path_expr,
        ))
        .map_with_span(|kind, span| Expr { kind, span: span_from_range(span) })
        .boxed();

        // Struct literal field: name: expr  or  name (shorthand)
        let struct_field = ident()
            .then(just(TokenKind::Colon).ignore_then(expr.clone()).or_not())
            .map(|(name, value)| {
                let shorthand = value.is_none();
                let value = value.unwrap_or_else(|| Expr {
                    kind: ExprKind::Path(Path {
                        segments: vec![PathSegment { ident: name.clone(), args: None }],
                        span: name.span,
                    }),
                    span: name.span,
                });
                FieldExpr { name, value, shorthand }
            });

        // Postfix operations as an enum for proper folding
        #[derive(Clone)]
        enum PostfixOp {
            Field(Ident),
            MethodCall(Ident, Option<GenericArgs>, Vec<Expr>),  // method, turbofish, args
            Call(Vec<Expr>),
            Index(Expr),
            Try,
            StructLit(Vec<FieldExpr>),  // Struct literal: { fields }
        }

        // Turbofish: ::<Type1, Type2>
        let turbofish = just(TokenKind::PathSep)
            .ignore_then(
                ty.clone()
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::Lt), just(TokenKind::Gt))
            )
            .map(|types| GenericArgs {
                args: types.into_iter().map(GenericArg::Type).collect(),
            });

        let postfix_op = choice((
            // Struct literal: { field: expr, ... } - must come before other braces
            struct_field.clone()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
                .map(PostfixOp::StructLit),
            // Method call or field access: .ident or .ident::<T>(args) or .ident(args) or .0 (tuple index)
            just(TokenKind::Dot)
                .ignore_then(
                    ident()
                        .or(select! { TokenKind::Int(n) => n }
                            .map_with_span(|n, span| Ident::new(n.to_string(), span_from_range(span))))
                )
                .then(turbofish.or_not())
                .then(
                    expr.clone()
                        .separated_by(just(TokenKind::Comma))
                        .allow_trailing()
                        .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
                        .or_not()
                )
                .map(|((field, turbo), args)| {
                    match args {
                        Some(args) => PostfixOp::MethodCall(field, turbo, args),
                        None if turbo.is_some() => PostfixOp::MethodCall(field, turbo, vec![]),
                        None => PostfixOp::Field(field),
                    }
                }),
            // Call: (args)
            expr.clone()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
                .map(PostfixOp::Call),
            // Index: [expr]
            expr.clone()
                .delimited_by(just(TokenKind::LBracket), just(TokenKind::RBracket))
                .map(PostfixOp::Index),
            // Try: ?
            just(TokenKind::Question).to(PostfixOp::Try),
        ));

        // Helper to fold postfix operations
        let fold_postfix = |e: Expr, op: PostfixOp| -> Expr {
            let span = e.span;
            let kind = match op {
                PostfixOp::Field(field) => ExprKind::Field {
                    expr: Box::new(e),
                    field,
                },
                PostfixOp::MethodCall(method, turbofish, args) => ExprKind::MethodCall {
                    receiver: Box::new(e),
                    method,
                    turbofish,
                    args,
                },
                PostfixOp::Call(args) => ExprKind::Call {
                    func: Box::new(e),
                    args,
                },
                PostfixOp::Index(index) => ExprKind::Index {
                    expr: Box::new(e),
                    index: Box::new(index),
                },
                PostfixOp::Try => ExprKind::Try(Box::new(e)),
                PostfixOp::StructLit(fields) => {
                    // The previous expr should be a path for the struct name
                    if let ExprKind::Path(path) = e.kind {
                        ExprKind::Struct { path, fields, rest: None }
                    } else {
                        // Invalid: non-path followed by struct literal
                        ExprKind::Struct {
                            path: Path {
                                segments: vec![],
                                span: e.span,
                            },
                            fields,
                            rest: None,
                        }
                    }
                }
            };
            Expr { kind, span }
        };

        // Postfix: calls, fields, indexing, struct literals
        let postfix = atom.clone()
            .then(postfix_op.repeated())
            .foldl(fold_postfix);

        // Unary prefix: handles -, !, *, and & (with optional mut)
        #[derive(Clone)]
        enum PrefixOp {
            Neg,
            Not,
            Deref,
            Ref(bool), // bool = mutable
        }

        let prefix_op = choice((
            just(TokenKind::Minus).to(PrefixOp::Neg),
            just(TokenKind::Not).to(PrefixOp::Not),
            just(TokenKind::Star).to(PrefixOp::Deref),
            just(TokenKind::And)
                .ignore_then(keyword(Keyword::Mut).or_not().map(|m| m.is_some()))
                .map(PrefixOp::Ref),
        ));

        // Helper for unary foldr
        let fold_unary = |op: PrefixOp, expr: Expr| -> Expr {
            let span = expr.span;
            let kind = match op {
                PrefixOp::Neg => ExprKind::Unary { op: UnaryOp::Neg, expr: Box::new(expr) },
                PrefixOp::Not => ExprKind::Unary { op: UnaryOp::Not, expr: Box::new(expr) },
                PrefixOp::Deref => ExprKind::Unary { op: UnaryOp::Deref, expr: Box::new(expr) },
                PrefixOp::Ref(mutable) => ExprKind::Ref { mutable, expr: Box::new(expr) },
            };
            Expr { span, kind }
        };

        let unary = prefix_op.clone()
            .repeated()
            .then(postfix.clone())
            .foldr(fold_unary);

        // Type cast: expr as Type
        let cast = unary.clone()
            .then(
                keyword(Keyword::As)
                    .ignore_then(ty.clone())
                    .repeated()
            )
            .foldl(|expr, ty| Expr {
                span: expr.span,
                kind: ExprKind::Cast { expr: Box::new(expr), ty },
            });

        // Binary operators with precedence
        let product = cast.clone()
            .then(
                choice((
                    just(TokenKind::Star).to(BinOp::Mul),
                    just(TokenKind::Slash).to(BinOp::Div),
                    just(TokenKind::Percent).to(BinOp::Rem),
                ))
                .then(cast.clone())
                .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let sum = product.clone()
            .then(
                choice((
                    just(TokenKind::Plus).to(BinOp::Add),
                    just(TokenKind::Minus).to(BinOp::Sub),
                ))
                .then(product.clone())
                .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        // Handle >> as either Shr token or two consecutive Gt tokens
        // (lexer may not combine them in generic contexts)
        let shr_op = just(TokenKind::Shr).to(BinOp::Shr)
            .or(just(TokenKind::Gt).ignore_then(just(TokenKind::Gt)).to(BinOp::Shr));

        let shift = sum.clone()
            .then(
                choice((
                    just(TokenKind::Shl).to(BinOp::Shl),
                    shr_op,
                ))
                .then(sum.clone())
                .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        // Bitwise AND: &
        let bit_and = shift.clone()
            .then(
                just(TokenKind::And).to(BinOp::BitAnd)
                    .then(shift.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        // Bitwise XOR: ^
        let bit_xor = bit_and.clone()
            .then(
                just(TokenKind::Caret).to(BinOp::BitXor)
                    .then(bit_and.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        // Bitwise OR: |
        let bit_or = bit_xor.clone()
            .then(
                just(TokenKind::Or).to(BinOp::BitOr)
                    .then(bit_xor.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let comparison = bit_or.clone()
            .then(
                choice((
                    just(TokenKind::EqEq).to(BinOp::Eq),
                    just(TokenKind::Ne).to(BinOp::Ne),
                    just(TokenKind::Lt).to(BinOp::Lt),
                    just(TokenKind::Le).to(BinOp::Le),
                    just(TokenKind::Gt).to(BinOp::Gt),
                    just(TokenKind::Ge).to(BinOp::Ge),
                ))
                .then(bit_or.clone())
                .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let logical_and = comparison.clone()
            .then(
                just(TokenKind::AndAnd).to(BinOp::And)
                    .then(comparison.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let logical_or = logical_and.clone()
            .then(
                just(TokenKind::OrOr).to(BinOp::Or)
                    .then(logical_and.clone())
                    .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        // Range expressions: a..b, a..=b, a..
        // Note: Prefix ranges (..b, ..) are parsed as atoms
        let logical_or_boxed = logical_or.clone().boxed();
        let range = logical_or_boxed.clone()
            .then(
                choice((
                    just(TokenKind::DotDotEq)
                        .ignore_then(logical_or_boxed.clone())
                        .map(|end| (true, Some(end))),
                    just(TokenKind::DotDot)
                        .ignore_then(logical_or_boxed.clone().or_not())
                        .map(|end| (false, end)),
                ))
                .or_not()
            )
            .map_with_span(|(start, range_end), span| {
                match range_end {
                    Some((inclusive, end)) => Expr {
                        kind: ExprKind::Range {
                            start: Some(Box::new(start)),
                            end: end.map(Box::new),
                            inclusive,
                        },
                        span: span_from_range(span),
                    },
                    None => start,
                }
            });

        // Assignment
        let assign = range.clone()
            .then(
                choice((
                    just(TokenKind::Eq).to(None),
                    just(TokenKind::PlusEq).to(Some(BinOp::Add)),
                    just(TokenKind::MinusEq).to(Some(BinOp::Sub)),
                    just(TokenKind::StarEq).to(Some(BinOp::Mul)),
                    just(TokenKind::SlashEq).to(Some(BinOp::Div)),
                    just(TokenKind::PercentEq).to(Some(BinOp::Rem)),
                    just(TokenKind::CaretEq).to(Some(BinOp::BitXor)),
                    just(TokenKind::OrEq).to(Some(BinOp::BitOr)),
                    just(TokenKind::AndEq).to(Some(BinOp::BitAnd)),
                    just(TokenKind::ShlEq).to(Some(BinOp::Shl)),
                    just(TokenKind::ShrEq).to(Some(BinOp::Shr)),
                ))
                .then(range.clone())
                .or_not()
            )
            .map(|(left, rhs)| {
                match rhs {
                    Some((None, right)) => Expr {
                        span: left.span.merge(right.span),
                        kind: ExprKind::Assign {
                            target: Box::new(left),
                            value: Box::new(right),
                        },
                    },
                    Some((Some(op), right)) => Expr {
                        span: left.span.merge(right.span),
                        kind: ExprKind::AssignOp {
                            op,
                            target: Box::new(left),
                            value: Box::new(right),
                        },
                    },
                    None => left,
                }
            });

        assign
    })
}

fn expr_parser() -> impl Parser<TokenKind, Expr, Error = ParseError> + Clone {
    let ty = type_parser();
    let pat = pattern_parser();
    let simple_pat = simple_pattern_parser();
    let balanced = balanced_tokens_parser();
    expr_parser_with(ty, pat, simple_pat, balanced)
}

// ============================================================================
// Block Parser
// ============================================================================

fn block_parser(
    expr: impl Parser<TokenKind, Expr, Error = ParseError> + Clone,
    ty: impl Parser<TokenKind, Type, Error = ParseError> + Clone,
    pat: impl Parser<TokenKind, Pattern, Error = ParseError> + Clone,
) -> impl Parser<TokenKind, Block, Error = ParseError> + Clone {
    let stmt = stmt_parser(expr, ty, pat);

    stmt.repeated()
        .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
        .map_with_span(|stmts, span| Block { stmts, span: span_from_range(span) })
}

/// Convenience function: creates a complete block parser with all dependencies
fn full_block_parser() -> impl Parser<TokenKind, Block, Error = ParseError> + Clone {
    let ty = type_parser();
    let pat = pattern_parser();
    let simple_pat = simple_pattern_parser();
    let balanced = balanced_tokens_parser();
    let expr = expr_parser_with(ty.clone(), pat.clone(), simple_pat, balanced);
    block_parser(expr, ty, pat)
}

fn stmt_parser(
    expr: impl Parser<TokenKind, Expr, Error = ParseError> + Clone,
    ty: impl Parser<TokenKind, Type, Error = ParseError> + Clone,
    pat: impl Parser<TokenKind, Pattern, Error = ParseError> + Clone,
) -> impl Parser<TokenKind, Stmt, Error = ParseError> + Clone {
    let let_stmt = keyword(Keyword::Let)
        .ignore_then(pat)
        .then(just(TokenKind::Colon).ignore_then(ty.clone()).or_not())
        .then(just(TokenKind::Eq).ignore_then(expr.clone()).or_not())
        .then_ignore(just(TokenKind::Semi))
        .map(|((pat, ty), init)| StmtKind::Let { pat, ty, init });

    // Local use statement: use path::to::thing;
    let use_stmt = keyword(Keyword::Use)
        .ignore_then(use_tree_parser())
        .then_ignore(just(TokenKind::Semi))
        .map_with_span(|tree, span| StmtKind::Item(Item {
            kind: ItemKind::Use(Use { tree, is_pub: false }),
            span: span_from_range(span),
        }));

    // Local const: const NAME: Type = value;
    let const_stmt = keyword(Keyword::Const)
        .ignore_then(ident())
        .then_ignore(just(TokenKind::Colon))
        .then(ty.clone())
        .then(just(TokenKind::Eq).ignore_then(expr.clone()).or_not())
        .then_ignore(just(TokenKind::Semi))
        .map_with_span(|((name, ty), value), span| StmtKind::Item(Item {
            kind: ItemKind::Const(Const { name, ty, value, is_pub: false }),
            span: span_from_range(span),
        }));

    let expr_stmt = expr.clone()
        .then(just(TokenKind::Semi).or_not())
        .map(|(e, semi)| {
            if semi.is_some() {
                StmtKind::Expr(e)
            } else {
                StmtKind::ExprNoSemi(e)
            }
        });

    let empty_stmt = just(TokenKind::Semi)
        .map(|_| StmtKind::Empty);

    choice((
        let_stmt,
        use_stmt,
        const_stmt,
        empty_stmt,
        expr_stmt,
    ))
    .map_with_span(|kind, span| Stmt { kind, span: span_from_range(span) })
}

// ============================================================================
// Item Parser
// ============================================================================

// Parser for skipping attributes like #[derive(...)]
fn skip_attributes() -> impl Parser<TokenKind, (), Error = ParseError> + Clone {
    // #[ident(...)] or #[ident = ...]
    // We use a simple approach: skip everything between #[ and ]
    // Handle nested brackets by counting
    let attr_content = none_of([TokenKind::RBracket])
        .repeated()
        .ignored();

    just(TokenKind::Pound)
        .ignore_then(just(TokenKind::LBracket))
        .ignore_then(attr_content)
        .ignore_then(just(TokenKind::RBracket))
        .ignored()
        .repeated()
        .ignored()
}

fn item_parser_with(
    ty: impl Parser<TokenKind, Type, Error = ParseError> + Clone + 'static,
    pat: impl Parser<TokenKind, Pattern, Error = ParseError> + Clone + 'static,
    block: impl Parser<TokenKind, Block, Error = ParseError> + Clone + 'static,
    expr: impl Parser<TokenKind, Expr, Error = ParseError> + Clone + 'static,
    balanced: impl Parser<TokenKind, (), Error = ParseError> + Clone + 'static,
    param: impl Parser<TokenKind, Param, Error = ParseError> + Clone + 'static,
    impl_item: impl Parser<TokenKind, ImplItem, Error = ParseError> + Clone + 'static,
    trait_item: impl Parser<TokenKind, TraitItem, Error = ParseError> + Clone + 'static,
) -> impl Parser<TokenKind, Item, Error = ParseError> + Clone {
    recursive(move |item| {
        let ty = ty.clone();
        let block = block.clone();
        let expr = expr.clone();
        let balanced = balanced.clone();
        let param = param.clone();
        let impl_item = impl_item.clone();
        let trait_item = trait_item.clone();
        let pat = pat.clone();

        // Skip any attributes before parsing the item
        let attrs = skip_attributes();

        let visibility = attrs.ignore_then(keyword(Keyword::Pub).or_not().map(|p| p.is_some()));

        // Function modifiers: unsafe and/or extern "C"
        let unsafe_modifier = keyword(Keyword::Unsafe).or_not().map(|u| u.is_some());
        let abi_specifier = keyword(Keyword::Extern)
            .ignore_then(select! { TokenKind::Str(s) => s }.or_not())
            .or_not();

        let function = visibility.clone()
            .then(unsafe_modifier)
            .then(abi_specifier)
            .then_ignore(keyword(Keyword::Fn))
            .then(ident())
            .then(generics_parser())
            .then(
                param.clone()
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
            )
            .then(just(TokenKind::Arrow).ignore_then(ty.clone()).or_not())
            .then(block.clone().or_not())
            .map(|(((((((is_pub, is_unsafe), _abi), name), generics), params), ret_type), body)| {
                ItemKind::Function(Function {
                    name,
                    generics,
                    params,
                    ret_type,
                    body,
                    is_async: false,
                    is_unsafe,
                    is_pub,
                })
            });

        // Tuple struct field: #[attr] pub Type or just Type
        let tuple_field = skip_attributes()
            .ignore_then(visibility.clone())
            .then(ty.clone())
            .map(|(is_pub, ty)| (is_pub, ty));

        let struct_item = visibility.clone()
            .then_ignore(keyword(Keyword::Struct))
            .then(ident())
            .then(generics_parser())
            .then(
                // Named struct: { field: Type, ... }
                field_parser()
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
                    .map(StructFields::Named)
                // Tuple struct: (Type, Type, ...);
                .or(
                    tuple_field
                        .separated_by(just(TokenKind::Comma))
                        .allow_trailing()
                        .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
                        .then_ignore(just(TokenKind::Semi))
                        .map(|fields| StructFields::Tuple(fields.into_iter().map(|(_, ty)| ty).collect()))
                )
                // Unit struct: ;
                .or(just(TokenKind::Semi).map(|_| StructFields::Unit))
            )
            .map(|(((is_pub, name), generics), fields)| {
                ItemKind::Struct(Struct { name, generics, fields, is_pub })
            });

        let enum_item = visibility.clone()
            .then_ignore(keyword(Keyword::Enum))
            .then(ident())
            .then(generics_parser())
            .then(
                variant_parser()
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
            )
            .map(|(((is_pub, name), generics), variants)| {
                ItemKind::Enum(Enum { name, generics, variants, is_pub })
            });

        let impl_block = keyword(Keyword::Impl)
            .ignore_then(generics_parser())
            .then(path_parser().then_ignore(keyword(Keyword::For)).or_not())
            .then(ty.clone())
            .then(
                impl_item.clone()
                    .repeated()
                    .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
            )
            .map(|(((generics, trait_), self_ty), items)| {
                ItemKind::Impl(Impl { generics, trait_, self_ty, items })
            });

        let trait_block = visibility.clone()
            .then_ignore(keyword(Keyword::Trait))
            .then(ident())
            .then(generics_parser())
            .then(
                trait_item.clone()
                    .repeated()
                    .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
            )
            .map(|(((is_pub, name), generics), items)| {
                ItemKind::Trait(Trait {
                    name,
                    generics,
                    bounds: vec![],
                    items,
                    is_pub,
                })
            });

        let use_item = visibility.clone()
            .then_ignore(keyword(Keyword::Use))
            .then(use_tree_parser())
            .then_ignore(just(TokenKind::Semi))
            .map(|(is_pub, tree)| ItemKind::Use(Use { tree, is_pub }));

        let const_item = visibility.clone()
            .then_ignore(keyword(Keyword::Const))
            .then(ident())
            .then_ignore(just(TokenKind::Colon))
            .then(ty.clone())
            .then(just(TokenKind::Eq).ignore_then(expr.clone()).or_not())
            .then_ignore(just(TokenKind::Semi))
            .map(|(((is_pub, name), t), value)| {
                ItemKind::Const(Const { name, ty: t, value, is_pub })
            });

        let static_item = visibility.clone()
            .then_ignore(keyword(Keyword::Static))
            .then(keyword(Keyword::Mut).or_not().map(|m| m.is_some()))
            .then(ident())
            .then_ignore(just(TokenKind::Colon))
            .then(ty.clone())
            .then(just(TokenKind::Eq).ignore_then(expr.clone()).or_not())
            .then_ignore(just(TokenKind::Semi))
            .map(|((((is_pub, is_mut), name), t), value)| {
                ItemKind::Static(Static { name, ty: t, value, is_mut, is_pub })
            });

        // Type alias: type Name = Type;
        let type_alias = visibility.clone()
            .then_ignore(keyword(Keyword::Type))
            .then(ident())
            .then(generics_parser())
            .then(just(TokenKind::Eq).ignore_then(ty.clone()).or_not())
            .then_ignore(just(TokenKind::Semi))
            .map(|(((is_pub, name), generics), ty)| {
                ItemKind::TypeAlias(TypeAlias { name, generics, ty, is_pub })
            });

        let mod_item = visibility.clone()
            .then_ignore(keyword(Keyword::Mod))
            .then(ident())
            .then(
                item.clone()
                    .repeated()
                    .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
                    .map(Some)
                .or(just(TokenKind::Semi).map(|_| None))
            )
            .map(|((is_pub, name), items)| {
                ItemKind::Module(Module { name, items, is_pub })
            });

        // macro_rules! name { ... } - skip the body (handles nested delimiters)
        // Use pre-created balanced tokens parser
        let macro_rules_item = just(TokenKind::Ident("macro_rules".to_string()))
            .ignore_then(just(TokenKind::Not))
            .ignore_then(ident())
            .then(
                just(TokenKind::LBrace)
                    .ignore_then(balanced.clone().repeated())
                    .then_ignore(just(TokenKind::RBrace))
                    .or(just(TokenKind::LParen)
                        .ignore_then(balanced.clone().repeated())
                        .then_ignore(just(TokenKind::RParen)))
            )
            .map(|(name, _)| ItemKind::MacroRules(MacroRules {
                name,
                rules: vec![],
            }));

        // extern "C" { fn declarations } - skip the body for now
        let extern_block = keyword(Keyword::Extern)
            .ignore_then(
                select! { TokenKind::Str(s) => s }.or_not()
            )
            .then(
                just(TokenKind::LBrace)
                    .ignore_then(balanced.clone().repeated())
                    .then_ignore(just(TokenKind::RBrace))
            )
            .map(|(abi, _)| ItemKind::ExternBlock(ExternBlock {
                abi,
                items: vec![],
            }));

        choice((
            macro_rules_item,
            extern_block,
            function,
            struct_item,
            enum_item,
            impl_block,
            trait_block,
            use_item,
            const_item,
            static_item,
            type_alias,
            mod_item,
        ))
        .map_with_span(|kind, span| Item { kind, span: span_from_range(span) })
        .recover_with(skip_then_retry_until([
            TokenKind::Keyword(Keyword::Fn),
            TokenKind::Keyword(Keyword::Struct),
            TokenKind::Keyword(Keyword::Enum),
            TokenKind::Keyword(Keyword::Impl),
            TokenKind::Keyword(Keyword::Trait),
            TokenKind::Keyword(Keyword::Use),
            TokenKind::Keyword(Keyword::Const),
            TokenKind::Keyword(Keyword::Static),
            TokenKind::Keyword(Keyword::Type),
            TokenKind::Keyword(Keyword::Extern),
            TokenKind::Keyword(Keyword::Mod),
            TokenKind::Keyword(Keyword::Pub),
        ]))
    })
}

fn item_parser() -> impl Parser<TokenKind, Item, Error = ParseError> + Clone {
    let ty = type_parser();
    let pat = pattern_parser();
    let simple_pat = simple_pattern_parser();
    let balanced = balanced_tokens_parser();
    let expr = expr_parser_with(ty.clone(), pat.clone(), simple_pat, balanced.clone());
    let block = block_parser(expr.clone(), ty.clone(), pat.clone());
    let param = param_parser_with(ty.clone(), pat.clone());
    let impl_item = impl_item_parser_with(ty.clone(), block.clone(), param.clone());
    let trait_item = trait_item_parser_with(ty.clone(), block.clone(), param.clone());
    item_parser_with(ty, pat, block, expr, balanced, param, impl_item, trait_item)
}

fn self_type() -> Type {
    Type {
        kind: TypeKind::Path(Path {
            segments: vec![PathSegment {
                ident: Ident::new("Self".to_string(), Span::dummy()),
                args: None,
            }],
            span: Span::dummy(),
        }),
        span: Span::dummy(),
    }
}

fn self_pattern(mutable: bool, span: Span) -> Pattern {
    Pattern {
        kind: PatternKind::Ident {
            mutable,
            by_ref: false,
            name: Ident::new("self".to_string(), span),
            subpat: None,
        },
        span,
    }
}

fn param_parser_with(
    ty: impl Parser<TokenKind, Type, Error = ParseError> + Clone,
    pat: impl Parser<TokenKind, Pattern, Error = ParseError> + Clone,
) -> impl Parser<TokenKind, Param, Error = ParseError> + Clone {
    // Self parameter variants: self, mut self, &self, &mut self
    let self_param = choice((
        // &mut self
        just(TokenKind::And)
            .ignore_then(keyword(Keyword::Mut))
            .ignore_then(keyword(Keyword::SelfLower))
            .map_with_span(|_, span| {
                let s = span_from_range(span);
                Param {
                    pattern: self_pattern(false, s),
                    ty: Type {
                        kind: TypeKind::Ref {
                            lifetime: None,
                            mutable: true,
                            inner: Box::new(self_type()),
                        },
                        span: s,
                    },
                    span: s,
                }
            }),
        // &self
        just(TokenKind::And)
            .ignore_then(keyword(Keyword::SelfLower))
            .map_with_span(|_, span| {
                let s = span_from_range(span);
                Param {
                    pattern: self_pattern(false, s),
                    ty: Type {
                        kind: TypeKind::Ref {
                            lifetime: None,
                            mutable: false,
                            inner: Box::new(self_type()),
                        },
                        span: s,
                    },
                    span: s,
                }
            }),
        // mut self
        keyword(Keyword::Mut)
            .ignore_then(keyword(Keyword::SelfLower))
            .map_with_span(|_, span| {
                let s = span_from_range(span);
                Param {
                    pattern: self_pattern(true, s),
                    ty: self_type(),
                    span: s,
                }
            }),
        // self
        keyword(Keyword::SelfLower)
            .map_with_span(|_, span| {
                let s = span_from_range(span);
                Param {
                    pattern: self_pattern(false, s),
                    ty: self_type(),
                    span: s,
                }
            }),
    ));

    // Regular parameter: pattern: type
    let regular_param = pat
        .then_ignore(just(TokenKind::Colon))
        .then(ty)
        .map_with_span(|(pattern, ty), span| Param {
            pattern,
            ty,
            span: span_from_range(span),
        });

    choice((self_param, regular_param))
}

fn param_parser() -> impl Parser<TokenKind, Param, Error = ParseError> + Clone {
    param_parser_with(type_parser(), pattern_parser())
}

fn field_parser() -> impl Parser<TokenKind, Field, Error = ParseError> + Clone {
    // Skip field attributes like #[serde(...)]
    skip_attributes()
        .ignore_then(keyword(Keyword::Pub).or_not())
        .then(ident())
        .then_ignore(just(TokenKind::Colon))
        .then(type_parser())
        .map_with_span(|((is_pub, name), ty), span| Field {
            name,
            ty,
            is_pub: is_pub.is_some(),
            span: span_from_range(span),
        })
}

fn variant_parser() -> impl Parser<TokenKind, Variant, Error = ParseError> + Clone {
    // Skip variant attributes like #[error(...)]
    skip_attributes()
        .ignore_then(ident())
        .then(
            field_parser()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
                .map(StructFields::Named)
            .or(
                // Tuple variant fields can have attributes like #[from]
                skip_attributes()
                    .ignore_then(type_parser())
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
                    .map(StructFields::Tuple)
            )
            .or_not()
            .map(|f| f.unwrap_or(StructFields::Unit))
        )
        .then(just(TokenKind::Eq).ignore_then(expr_parser()).or_not())
        .map_with_span(|((name, fields), discriminant), span| Variant {
            name,
            fields,
            discriminant,
            span: span_from_range(span),
        })
}

fn generics_parser() -> impl Parser<TokenKind, Generics, Error = ParseError> + Clone {
    let type_param = ident()
        .then(just(TokenKind::Colon).ignore_then(type_bounds()).or_not())
        .map(|(name, bounds)| GenericParam::Type {
            name,
            bounds: bounds.unwrap_or_default(),
            default: None,
        });

    let lifetime_param = lifetime()
        .map(|name| GenericParam::Lifetime { name, bounds: vec![] });

    choice((lifetime_param, type_param))
        .separated_by(just(TokenKind::Comma))
        .allow_trailing()
        .delimited_by(just(TokenKind::Lt), just(TokenKind::Gt))
        .or_not()
        .map(|params| Generics {
            params: params.unwrap_or_default(),
            where_clause: None,
        })
}

fn impl_item_parser_with(
    ty: impl Parser<TokenKind, Type, Error = ParseError> + Clone,
    block: impl Parser<TokenKind, Block, Error = ParseError> + Clone,
    param: impl Parser<TokenKind, Param, Error = ParseError> + Clone,
) -> impl Parser<TokenKind, ImplItem, Error = ParseError> + Clone {
    let function = skip_attributes()
        .ignore_then(keyword(Keyword::Pub).or_not().map(|p| p.is_some()))
        .then_ignore(keyword(Keyword::Fn))
        .then(ident())
        .then(generics_parser())
        .then(
            param
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
        )
        .then(just(TokenKind::Arrow).ignore_then(ty).or_not())
        .then(block)
        .map(|(((((is_pub, name), generics), params), ret_type), body)| {
            ImplItemKind::Function(Function {
                name,
                generics,
                params,
                ret_type,
                body: Some(body),
                is_async: false,
                is_unsafe: false,
                is_pub,
            })
        });

    function.map_with_span(|kind, span| ImplItem { kind, span: span_from_range(span) })
}

fn impl_item_parser() -> impl Parser<TokenKind, ImplItem, Error = ParseError> + Clone {
    impl_item_parser_with(type_parser(), full_block_parser(), param_parser())
}

fn trait_item_parser_with(
    ty: impl Parser<TokenKind, Type, Error = ParseError> + Clone,
    block: impl Parser<TokenKind, Block, Error = ParseError> + Clone,
    param: impl Parser<TokenKind, Param, Error = ParseError> + Clone,
) -> impl Parser<TokenKind, TraitItem, Error = ParseError> + Clone {
    let function = skip_attributes()
        .ignore_then(keyword(Keyword::Fn))
        .ignore_then(ident())
        .then(generics_parser())
        .then(
            param
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
        )
        .then(just(TokenKind::Arrow).ignore_then(ty).or_not())
        .then(
            block.map(Some)
                .or(just(TokenKind::Semi).map(|_| None))
        )
        .map(|((((name, generics), params), ret_type), body)| {
            TraitItemKind::Function(Function {
                name,
                generics,
                params,
                ret_type,
                body,
                is_async: false,
                is_unsafe: false,
                is_pub: false,
            })
        });

    function.map_with_span(|kind, span| TraitItem { kind, span: span_from_range(span) })
}

fn trait_item_parser() -> impl Parser<TokenKind, TraitItem, Error = ParseError> + Clone {
    trait_item_parser_with(type_parser(), full_block_parser(), param_parser())
}

fn use_tree_parser() -> impl Parser<TokenKind, UseTree, Error = ParseError> + Clone {
    recursive(|tree| {
        let path = path_parser();

        // Parse: path (::* | ::{...} | as ident)?
        path.then(
            choice((
                // ::* (glob import)
                just(TokenKind::PathSep)
                    .ignore_then(just(TokenKind::Star))
                    .map(|_| ("glob", None::<Ident>, Vec::new())),
                // ::{...} (group import)
                just(TokenKind::PathSep)
                    .ignore_then(
                        tree.clone()
                            .separated_by(just(TokenKind::Comma))
                            .allow_trailing()
                            .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
                    )
                    .map(|items| ("group", None, items)),
                // as ident (rename)
                keyword(Keyword::As)
                    .ignore_then(ident())
                    .map(|alias| ("alias", Some(alias), Vec::new())),
            ))
            .or_not()
        )
        .map(|(path, suffix)| {
            match suffix {
                Some(("glob", _, _)) => UseTree::Path(path, Some(Box::new(UseTree::Glob))),
                Some(("group", _, items)) => UseTree::Path(path, Some(Box::new(UseTree::Group(items)))),
                Some(("alias", Some(alias), _)) => {
                    // use foo::bar as baz -> Alias(bar, baz)
                    let name = path.segments.last()
                        .map(|s| s.ident.clone())
                        .unwrap_or_else(|| Ident::new("".to_string(), Span::dummy()));
                    UseTree::Path(
                        Path {
                            segments: path.segments[..path.segments.len().saturating_sub(1)].to_vec(),
                            span: path.span,
                        },
                        Some(Box::new(UseTree::Alias(name, alias)))
                    )
                }
                _ => UseTree::Path(path, None),
            }
        })
    })
}

// ============================================================================
// Source File Parser
// ============================================================================

fn source_file_parser() -> impl Parser<TokenKind, SourceFile, Error = ParseError> {
    // Create all parsers ONCE at the top level to avoid nested construction
    let ty = type_parser();
    let pat = pattern_parser();
    let simple_pat = simple_pattern_parser();
    let balanced = balanced_tokens_parser();
    let expr = expr_parser_with(ty.clone(), pat.clone(), simple_pat, balanced.clone());
    let block = block_parser(expr.clone(), ty.clone(), pat.clone());
    let param = param_parser_with(ty.clone(), pat.clone());
    let impl_item = impl_item_parser_with(ty.clone(), block.clone(), param.clone());
    let trait_item = trait_item_parser_with(ty.clone(), block.clone(), param.clone());
    let item = item_parser_with(ty, pat, block, expr, balanced, param, impl_item, trait_item);

    item.repeated()
        .then_ignore(end())
        .map_with_span(|items, span| SourceFile {
            items,
            span: span_from_range(span),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let source = "fn main() { 42 }";
        let (ast, errors) = parse(source);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        assert!(ast.is_some());
        let ast = ast.unwrap();
        assert_eq!(ast.items.len(), 1);
    }

    #[test]
    fn test_parse_struct() {
        let source = "struct Point { x: i64, y: i64 }";
        let (ast, errors) = parse(source);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        assert!(ast.is_some());
    }

    #[test]
    fn test_parse_expression() {
        let source = "fn f() { 1 + 2 * 3 }";
        let (ast, errors) = parse(source);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        assert!(ast.is_some());
    }

    #[test]
    fn test_error_recovery() {
        let source = "fn main( { } fn other() { }";
        let (ast, errors) = parse(source);
        // Should recover and parse second function
        assert!(!errors.is_empty());
        assert!(ast.is_some());
    }
}
