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

        let segment = ident()
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
            .ignore_then(type_bounds())
            .map(TypeKind::ImplTrait);

        let dyn_trait = keyword(Keyword::Dyn)
            .ignore_then(type_bounds())
            .map(TypeKind::DynTrait);

        choice((
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

// ============================================================================
// Path Parser
// ============================================================================

fn path_parser() -> impl Parser<TokenKind, Path, Error = ParseError> + Clone {
    // Simple path parser for expressions - no generic args to avoid mutual recursion
    // Turbofish syntax (e.g., Vec::<i32>::new()) can be handled separately if needed
    let segment = ident()
        .map(|ident| PathSegment { ident, args: None });

    segment
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
                field_pat
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .then(just(TokenKind::DotDot).or_not())
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
// Expression Parser
// ============================================================================

fn expr_parser() -> impl Parser<TokenKind, Expr, Error = ParseError> + Clone {
    recursive(|expr| {
        let block = block_parser(expr.clone());

        // Atoms
        let lit_expr = lit().map(ExprKind::Lit);
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

        let if_expr = keyword(Keyword::If)
            .ignore_then(expr.clone())
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

        let match_arm = pattern_parser()
            .then(just(TokenKind::Keyword(Keyword::If)).ignore_then(expr.clone()).or_not())
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
                match_arm
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
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
            .ignore_then(expr.clone())
            .then(block.clone())
            .map(|(cond, body)| ExprKind::While {
                label: None,
                cond: Box::new(cond),
                body,
            });

        let for_expr = keyword(Keyword::For)
            .ignore_then(pattern_parser())
            .then_ignore(keyword(Keyword::In))
            .then(expr.clone())
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

        let closure = keyword(Keyword::Move).or_not()
            .then(
                pattern_parser()
                    .then(just(TokenKind::Colon).ignore_then(type_parser()).or_not())
                    .map(|(pat, ty)| ClosureParam { pattern: pat, ty })
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::Or), just(TokenKind::Or))
                    .or(just(TokenKind::OrOr).map(|_| Vec::new()))
            )
            .then(just(TokenKind::Arrow).ignore_then(type_parser()).or_not())
            .then(expr.clone())
            .map(|(((is_move, params), ret_type), body)| ExprKind::Closure {
                is_move: is_move.is_some(),
                params,
                ret_type,
                body: Box::new(body),
            });

        let block_expr = block.clone().map(ExprKind::Block);

        // Primary expression
        let atom = choice((
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
            block_expr,
            tuple_expr,
            array_expr,
            self_expr,  // must come before path_expr
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
            MethodCall(Ident, Vec<Expr>),
            Call(Vec<Expr>),
            Index(Expr),
            Try,
            StructLit(Vec<FieldExpr>),  // Struct literal: { fields }
        }

        let postfix_op = choice((
            // Struct literal: { field: expr, ... } - must come before other braces
            struct_field.clone()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
                .map(PostfixOp::StructLit),
            // Method call or field access: .ident or .ident(args) or .0 (tuple index)
            just(TokenKind::Dot)
                .ignore_then(
                    ident()
                        .or(select! { TokenKind::Int(n) => n }
                            .map_with_span(|n, span| Ident::new(n.to_string(), span_from_range(span))))
                )
                .then(
                    expr.clone()
                        .separated_by(just(TokenKind::Comma))
                        .allow_trailing()
                        .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
                        .or_not()
                )
                .map(|(field, args)| {
                    match args {
                        Some(args) => PostfixOp::MethodCall(field, args),
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

        // Postfix: calls, fields, indexing, struct literals
        let postfix = atom.clone()
            .then(postfix_op.repeated())
            .foldl(|e, op| {
                let span = e.span;
                let kind = match op {
                    PostfixOp::Field(field) => ExprKind::Field {
                        expr: Box::new(e),
                        field,
                    },
                    PostfixOp::MethodCall(method, args) => ExprKind::MethodCall {
                        receiver: Box::new(e),
                        method,
                        turbofish: None,
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
                            // Just return the fields as struct with dummy path
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
            });

        // Unary prefix
        let unary = choice((
            just(TokenKind::Minus).to(UnaryOp::Neg),
            just(TokenKind::Not).to(UnaryOp::Not),
            just(TokenKind::Star).to(UnaryOp::Deref),
        ))
        .repeated()
        .then(postfix)
        .foldr(|op, expr| Expr {
            span: expr.span,
            kind: ExprKind::Unary { op, expr: Box::new(expr) },
        });

        // Binary operators with precedence
        let product = unary.clone()
            .then(
                choice((
                    just(TokenKind::Star).to(BinOp::Mul),
                    just(TokenKind::Slash).to(BinOp::Div),
                    just(TokenKind::Percent).to(BinOp::Rem),
                ))
                .then(unary.clone())
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

        let shift = sum.clone()
            .then(
                choice((
                    just(TokenKind::Shl).to(BinOp::Shl),
                    just(TokenKind::Shr).to(BinOp::Shr),
                ))
                .then(sum.clone())
                .repeated()
            )
            .foldl(|a, (op, b)| Expr {
                span: a.span.merge(b.span),
                kind: ExprKind::Binary { op, left: Box::new(a), right: Box::new(b) },
            });

        let comparison = shift.clone()
            .then(
                choice((
                    just(TokenKind::EqEq).to(BinOp::Eq),
                    just(TokenKind::Ne).to(BinOp::Ne),
                    just(TokenKind::Lt).to(BinOp::Lt),
                    just(TokenKind::Le).to(BinOp::Le),
                    just(TokenKind::Gt).to(BinOp::Gt),
                    just(TokenKind::Ge).to(BinOp::Ge),
                ))
                .then(shift.clone())
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

// ============================================================================
// Block Parser
// ============================================================================

fn block_parser(expr: impl Parser<TokenKind, Expr, Error = ParseError> + Clone) -> impl Parser<TokenKind, Block, Error = ParseError> + Clone {
    let stmt = stmt_parser(expr);

    stmt.repeated()
        .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
        .map_with_span(|stmts, span| Block { stmts, span: span_from_range(span) })
}

fn stmt_parser(expr: impl Parser<TokenKind, Expr, Error = ParseError> + Clone) -> impl Parser<TokenKind, Stmt, Error = ParseError> + Clone {
    let let_stmt = keyword(Keyword::Let)
        .ignore_then(pattern_parser())
        .then(just(TokenKind::Colon).ignore_then(type_parser()).or_not())
        .then(just(TokenKind::Eq).ignore_then(expr.clone()).or_not())
        .then_ignore(just(TokenKind::Semi))
        .map(|((pat, ty), init)| StmtKind::Let { pat, ty, init });

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
        empty_stmt,
        expr_stmt,
    ))
    .map_with_span(|kind, span| Stmt { kind, span: span_from_range(span) })
}

// ============================================================================
// Item Parser
// ============================================================================

fn item_parser() -> impl Parser<TokenKind, Item, Error = ParseError> + Clone {
    recursive(|item| {
        let visibility = keyword(Keyword::Pub).or_not().map(|p| p.is_some());

        let function = visibility.clone()
            .then_ignore(keyword(Keyword::Fn))
            .then(ident())
            .then(generics_parser())
            .then(
                param_parser()
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
            )
            .then(just(TokenKind::Arrow).ignore_then(type_parser()).or_not())
            .then(block_parser(expr_parser()).or_not())
            .map(|(((((is_pub, name), generics), params), ret_type), body)| {
                ItemKind::Function(Function {
                    name,
                    generics,
                    params,
                    ret_type,
                    body,
                    is_async: false,
                    is_unsafe: false,
                    is_pub,
                })
            });

        let struct_item = visibility.clone()
            .then_ignore(keyword(Keyword::Struct))
            .then(ident())
            .then(generics_parser())
            .then(
                field_parser()
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
                    .map(StructFields::Named)
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

        let impl_item = keyword(Keyword::Impl)
            .ignore_then(generics_parser())
            .then(path_parser().then_ignore(keyword(Keyword::For)).or_not())
            .then(type_parser())
            .then(
                impl_item_parser()
                    .repeated()
                    .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
            )
            .map(|(((generics, trait_), self_ty), items)| {
                ItemKind::Impl(Impl { generics, trait_, self_ty, items })
            });

        let trait_item = visibility.clone()
            .then_ignore(keyword(Keyword::Trait))
            .then(ident())
            .then(generics_parser())
            .then(
                trait_item_parser()
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
            .then(type_parser())
            .then(just(TokenKind::Eq).ignore_then(expr_parser()).or_not())
            .then_ignore(just(TokenKind::Semi))
            .map(|(((is_pub, name), ty), value)| {
                ItemKind::Const(Const { name, ty, value, is_pub })
            });

        let static_item = visibility.clone()
            .then_ignore(keyword(Keyword::Static))
            .then(keyword(Keyword::Mut).or_not().map(|m| m.is_some()))
            .then(ident())
            .then_ignore(just(TokenKind::Colon))
            .then(type_parser())
            .then(just(TokenKind::Eq).ignore_then(expr_parser()).or_not())
            .then_ignore(just(TokenKind::Semi))
            .map(|((((is_pub, is_mut), name), ty), value)| {
                ItemKind::Static(Static { name, ty, value, is_mut, is_pub })
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

        choice((
            function,
            struct_item,
            enum_item,
            impl_item,
            trait_item,
            use_item,
            const_item,
            static_item,
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
            TokenKind::Keyword(Keyword::Mod),
            TokenKind::Keyword(Keyword::Pub),
        ]))
    })
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

fn param_parser() -> impl Parser<TokenKind, Param, Error = ParseError> + Clone {
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
    let regular_param = pattern_parser()
        .then_ignore(just(TokenKind::Colon))
        .then(type_parser())
        .map_with_span(|(pattern, ty), span| Param {
            pattern,
            ty,
            span: span_from_range(span),
        });

    choice((self_param, regular_param))
}

fn field_parser() -> impl Parser<TokenKind, Field, Error = ParseError> + Clone {
    keyword(Keyword::Pub).or_not()
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
    ident()
        .then(
            field_parser()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
                .map(StructFields::Named)
            .or(
                type_parser()
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

fn impl_item_parser() -> impl Parser<TokenKind, ImplItem, Error = ParseError> + Clone {
    let function = keyword(Keyword::Fn)
        .ignore_then(ident())
        .then(generics_parser())
        .then(
            param_parser()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
        )
        .then(just(TokenKind::Arrow).ignore_then(type_parser()).or_not())
        .then(block_parser(expr_parser()))
        .map(|((((name, generics), params), ret_type), body)| {
            ImplItemKind::Function(Function {
                name,
                generics,
                params,
                ret_type,
                body: Some(body),
                is_async: false,
                is_unsafe: false,
                is_pub: false,
            })
        });

    function.map_with_span(|kind, span| ImplItem { kind, span: span_from_range(span) })
}

fn trait_item_parser() -> impl Parser<TokenKind, TraitItem, Error = ParseError> + Clone {
    let function = keyword(Keyword::Fn)
        .ignore_then(ident())
        .then(generics_parser())
        .then(
            param_parser()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .delimited_by(just(TokenKind::LParen), just(TokenKind::RParen))
        )
        .then(just(TokenKind::Arrow).ignore_then(type_parser()).or_not())
        .then(
            block_parser(expr_parser()).map(Some)
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

fn use_tree_parser() -> impl Parser<TokenKind, UseTree, Error = ParseError> + Clone {
    recursive(|tree| {
        let path = path_parser();

        path.then(
            just(TokenKind::PathSep)
                .ignore_then(
                    just(TokenKind::Star).map(|_| UseTree::Glob)
                        .or(tree.clone()
                            .separated_by(just(TokenKind::Comma))
                            .allow_trailing()
                            .delimited_by(just(TokenKind::LBrace), just(TokenKind::RBrace))
                            .map(UseTree::Group))
                )
                .or_not()
        )
        .map(|(path, subtree)| {
            match subtree {
                Some(sub) => UseTree::Path(path, Some(Box::new(sub))),
                None => UseTree::Path(path, None),
            }
        })
    })
}

// ============================================================================
// Source File Parser
// ============================================================================

fn source_file_parser() -> impl Parser<TokenKind, SourceFile, Error = ParseError> {
    item_parser()
        .repeated()
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
