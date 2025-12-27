//! Bolt Lexer - wrapper around rustc_lexer
//!
//! Uses the official Rust tokenizer with keyword recognition and literal parsing.

pub mod token;

pub use token::{Keyword, Span, Token, TokenKind};

use rustc_lexer::LiteralKind;

/// Lexer error
#[derive(Debug, Clone)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

/// Lexer wrapping rustc_lexer with keyword recognition
pub struct Lexer<'a> {
    source: &'a str,
    pos: usize,
    errors: Vec<LexError>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            pos: 0,
            errors: Vec::new(),
        }
    }

    /// Get remaining source
    fn remaining(&self) -> &'a str {
        &self.source[self.pos..]
    }

    /// Tokenize entire source into a vector
    pub fn tokenize(source: &str) -> (Vec<Token>, Vec<LexError>) {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();

        loop {
            let tok = lexer.next_token();
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }

        (tokens, lexer.errors)
    }

    /// Get next token
    pub fn next_token(&mut self) -> Token {
        // Skip whitespace and comments
        self.skip_trivia();

        if self.pos >= self.source.len() {
            return Token::new(TokenKind::Eof, Span::new(self.pos as u32, self.pos as u32));
        }

        let start = self.pos;
        let remaining = self.remaining();
        let rustc_tok = rustc_lexer::first_token(remaining);
        let len = rustc_tok.len;
        let text = &remaining[..len];
        self.pos += len;
        let span = Span::new(start as u32, self.pos as u32);

        let kind = self.convert_token(rustc_tok.kind, text, span);
        Token::new(kind, span)
    }

    /// Skip whitespace and comments
    fn skip_trivia(&mut self) {
        loop {
            if self.pos >= self.source.len() {
                break;
            }

            let remaining = self.remaining();
            let tok = rustc_lexer::first_token(remaining);

            match tok.kind {
                rustc_lexer::TokenKind::Whitespace => {
                    self.pos += tok.len;
                }
                rustc_lexer::TokenKind::LineComment => {
                    self.pos += tok.len;
                }
                rustc_lexer::TokenKind::BlockComment { terminated } => {
                    if !terminated {
                        self.errors.push(LexError {
                            message: "unterminated block comment".to_string(),
                            span: Span::new(self.pos as u32, (self.pos + tok.len) as u32),
                        });
                    }
                    self.pos += tok.len;
                }
                _ => break,
            }
        }
    }

    /// Convert rustc_lexer token to our token
    fn convert_token(&mut self, kind: rustc_lexer::TokenKind, text: &str, span: Span) -> TokenKind {
        use rustc_lexer::TokenKind as RK;

        match kind {
            // Identifiers and keywords
            RK::Ident => {
                if let Some(kw) = Keyword::from_str(text) {
                    TokenKind::Keyword(kw)
                } else if text == "_" {
                    TokenKind::Underscore
                } else {
                    TokenKind::Ident(text.to_string())
                }
            }
            RK::RawIdent => {
                // r#ident - strip the r# prefix
                TokenKind::Ident(text[2..].to_string())
            }

            // Literals
            RK::Literal { kind, suffix_start } => {
                self.parse_literal(kind, text, suffix_start, span)
            }

            // Lifetime or char
            RK::Lifetime { starts_with_number } => {
                if starts_with_number {
                    self.errors.push(LexError {
                        message: "lifetime cannot start with a number".to_string(),
                        span,
                    });
                }
                // 'lifetime - strip the leading quote
                TokenKind::Lifetime(text[1..].to_string())
            }

            // Punctuation - single char
            RK::Semi => TokenKind::Semi,
            RK::Comma => TokenKind::Comma,
            RK::Dot => TokenKind::Dot,
            RK::OpenParen => TokenKind::LParen,
            RK::CloseParen => TokenKind::RParen,
            RK::OpenBrace => TokenKind::LBrace,
            RK::CloseBrace => TokenKind::RBrace,
            RK::OpenBracket => TokenKind::LBracket,
            RK::CloseBracket => TokenKind::RBracket,
            RK::At => TokenKind::At,
            RK::Pound => TokenKind::Pound,
            RK::Tilde => TokenKind::Tilde,
            RK::Question => TokenKind::Question,
            RK::Colon => TokenKind::Colon,
            RK::Dollar => TokenKind::Dollar,
            RK::Eq => TokenKind::Eq,
            RK::Not => TokenKind::Not,
            RK::Lt => TokenKind::Lt,
            RK::Gt => TokenKind::Gt,
            RK::Minus => TokenKind::Minus,
            RK::And => TokenKind::And,
            RK::Or => TokenKind::Or,
            RK::Plus => TokenKind::Plus,
            RK::Star => TokenKind::Star,
            RK::Slash => TokenKind::Slash,
            RK::Caret => TokenKind::Caret,
            RK::Percent => TokenKind::Percent,

            // These shouldn't appear after skip_trivia
            RK::Whitespace | RK::LineComment | RK::BlockComment { .. } => {
                // Skip and get next token
                self.next_token().kind
            }

            RK::Unknown => {
                let c = text.chars().next().unwrap_or('?');
                self.errors.push(LexError {
                    message: format!("unexpected character: '{}'", c),
                    span,
                });
                TokenKind::Unknown(c)
            }
        }
    }

    /// Parse a literal token
    fn parse_literal(&mut self, kind: LiteralKind, text: &str, suffix_start: usize, span: Span) -> TokenKind {
        let _suffix = &text[suffix_start..];
        let main = &text[..suffix_start];

        match kind {
            LiteralKind::Int { base, empty_int } => {
                if empty_int {
                    self.errors.push(LexError {
                        message: "empty integer literal".to_string(),
                        span,
                    });
                    return TokenKind::Int(0);
                }
                self.parse_int(main, base, span)
            }

            LiteralKind::Float { base, empty_exponent } => {
                if empty_exponent {
                    self.errors.push(LexError {
                        message: "empty exponent in float literal".to_string(),
                        span,
                    });
                }
                if base != rustc_lexer::Base::Decimal {
                    self.errors.push(LexError {
                        message: "non-decimal float literal".to_string(),
                        span,
                    });
                }
                self.parse_float(main, span)
            }

            LiteralKind::Char { terminated } => {
                if !terminated {
                    self.errors.push(LexError {
                        message: "unterminated character literal".to_string(),
                        span,
                    });
                    return TokenKind::Char('\0');
                }
                self.parse_char(main, span)
            }

            LiteralKind::Byte { terminated } => {
                if !terminated {
                    self.errors.push(LexError {
                        message: "unterminated byte literal".to_string(),
                        span,
                    });
                    return TokenKind::Byte(0);
                }
                self.parse_byte(main, span)
            }

            LiteralKind::Str { terminated } => {
                if !terminated {
                    self.errors.push(LexError {
                        message: "unterminated string literal".to_string(),
                        span,
                    });
                    return TokenKind::Str(String::new());
                }
                self.parse_string(main, span)
            }

            LiteralKind::ByteStr { terminated } => {
                if !terminated {
                    self.errors.push(LexError {
                        message: "unterminated byte string literal".to_string(),
                        span,
                    });
                    return TokenKind::ByteStr(Vec::new());
                }
                self.parse_byte_string(main, span)
            }

            LiteralKind::RawStr { n_hashes: _, started, terminated } => {
                if !started {
                    self.errors.push(LexError {
                        message: "invalid raw string starter".to_string(),
                        span,
                    });
                    return TokenKind::RawStr(String::new());
                }
                if !terminated {
                    self.errors.push(LexError {
                        message: "unterminated raw string".to_string(),
                        span,
                    });
                    return TokenKind::RawStr(String::new());
                }
                self.parse_raw_string(main)
            }

            LiteralKind::RawByteStr { n_hashes: _, started, terminated } => {
                if !started || !terminated {
                    self.errors.push(LexError {
                        message: "invalid raw byte string".to_string(),
                        span,
                    });
                    return TokenKind::ByteStr(Vec::new());
                }
                self.parse_raw_byte_string(main)
            }
        }
    }

    fn parse_int(&mut self, text: &str, base: rustc_lexer::Base, span: Span) -> TokenKind {
        // Remove underscores
        let clean: String = text.chars().filter(|&c| c != '_').collect();

        // Strip prefix for non-decimal
        let (digits, radix) = match base {
            rustc_lexer::Base::Binary => (&clean[2..], 2),
            rustc_lexer::Base::Octal => (&clean[2..], 8),
            rustc_lexer::Base::Hexadecimal => (&clean[2..], 16),
            rustc_lexer::Base::Decimal => (clean.as_str(), 10),
        };

        match i128::from_str_radix(digits, radix) {
            Ok(n) => TokenKind::Int(n),
            Err(_) => {
                self.errors.push(LexError {
                    message: format!("invalid integer literal: {}", text),
                    span,
                });
                TokenKind::Int(0)
            }
        }
    }

    fn parse_float(&mut self, text: &str, span: Span) -> TokenKind {
        let clean: String = text.chars().filter(|&c| c != '_').collect();
        match clean.parse::<f64>() {
            Ok(n) => TokenKind::Float(n),
            Err(_) => {
                self.errors.push(LexError {
                    message: format!("invalid float literal: {}", text),
                    span,
                });
                TokenKind::Float(0.0)
            }
        }
    }

    fn parse_char(&mut self, text: &str, span: Span) -> TokenKind {
        // text is 'x' including quotes
        let inner = &text[1..text.len()-1];
        match self.unescape_char(inner) {
            Ok(c) => TokenKind::Char(c),
            Err(e) => {
                self.errors.push(LexError {
                    message: e,
                    span,
                });
                TokenKind::Char('\0')
            }
        }
    }

    fn parse_byte(&mut self, text: &str, span: Span) -> TokenKind {
        // text is b'x' including quotes
        let inner = &text[2..text.len()-1];
        match self.unescape_char(inner) {
            Ok(c) if c.is_ascii() => TokenKind::Byte(c as u8),
            Ok(_) => {
                self.errors.push(LexError {
                    message: "non-ascii character in byte literal".to_string(),
                    span,
                });
                TokenKind::Byte(0)
            }
            Err(e) => {
                self.errors.push(LexError {
                    message: e,
                    span,
                });
                TokenKind::Byte(0)
            }
        }
    }

    fn parse_string(&mut self, text: &str, _span: Span) -> TokenKind {
        // text is "..." including quotes
        let inner = &text[1..text.len()-1];
        TokenKind::Str(self.unescape_string(inner))
    }

    fn parse_byte_string(&mut self, text: &str, _span: Span) -> TokenKind {
        // text is b"..." including quotes
        let inner = &text[2..text.len()-1];
        TokenKind::ByteStr(self.unescape_string(inner).into_bytes())
    }

    fn parse_raw_string(&mut self, text: &str) -> TokenKind {
        // text is r#"..."# - find the content between quotes
        let start = text.find('"').unwrap_or(1) + 1;
        let end = text.rfind('"').unwrap_or(text.len());
        TokenKind::RawStr(text[start..end].to_string())
    }

    fn parse_raw_byte_string(&mut self, text: &str) -> TokenKind {
        // text is br#"..."#
        let start = text.find('"').unwrap_or(2) + 1;
        let end = text.rfind('"').unwrap_or(text.len());
        TokenKind::ByteStr(text[start..end].as_bytes().to_vec())
    }

    fn unescape_char(&self, s: &str) -> Result<char, String> {
        let mut chars = s.chars();
        match chars.next() {
            None => Err("empty character literal".to_string()),
            Some('\\') => {
                match chars.next() {
                    Some('n') => Ok('\n'),
                    Some('r') => Ok('\r'),
                    Some('t') => Ok('\t'),
                    Some('\\') => Ok('\\'),
                    Some('\'') => Ok('\''),
                    Some('"') => Ok('"'),
                    Some('0') => Ok('\0'),
                    Some('x') => {
                        // \xNN
                        let hex: String = chars.take(2).collect();
                        u8::from_str_radix(&hex, 16)
                            .map(|b| b as char)
                            .map_err(|_| format!("invalid hex escape: \\x{}", hex))
                    }
                    Some('u') => {
                        // \u{NNNN}
                        if chars.next() != Some('{') {
                            return Err("expected { in unicode escape".to_string());
                        }
                        let hex: String = chars.take_while(|&c| c != '}').collect();
                        u32::from_str_radix(&hex, 16)
                            .ok()
                            .and_then(char::from_u32)
                            .ok_or_else(|| format!("invalid unicode escape: \\u{{{}}}", hex))
                    }
                    Some(c) => Err(format!("unknown escape: \\{}", c)),
                    None => Err("incomplete escape sequence".to_string()),
                }
            }
            Some(c) => {
                if chars.next().is_some() {
                    Err("character literal too long".to_string())
                } else {
                    Ok(c)
                }
            }
        }
    }

    fn unescape_string(&self, s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('\'') => result.push('\''),
                    Some('"') => result.push('"'),
                    Some('0') => result.push('\0'),
                    Some('x') => {
                        let h1 = chars.next().unwrap_or('0');
                        let h2 = chars.next().unwrap_or('0');
                        let hex = format!("{}{}", h1, h2);
                        if let Ok(b) = u8::from_str_radix(&hex, 16) {
                            result.push(b as char);
                        }
                    }
                    Some('u') => {
                        if chars.next() == Some('{') {
                            let hex: String = chars.by_ref().take_while(|&c| c != '}').collect();
                            if let Some(c) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                                result.push(c);
                            }
                        }
                    }
                    Some('\n') => {
                        // Line continuation - skip whitespace
                        while chars.peek().map(|c| c.is_whitespace()).unwrap_or(false) {
                            chars.next();
                        }
                    }
                    Some(c) => {
                        result.push('\\');
                        result.push(c);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }
        result
    }

    /// Peek at next token without consuming
    pub fn peek(&mut self) -> Token {
        let saved_pos = self.pos;
        let saved_errors_len = self.errors.len();
        let tok = self.next_token();
        self.pos = saved_pos;
        self.errors.truncate(saved_errors_len);
        tok
    }

    /// Get all errors
    pub fn errors(&self) -> &[LexError] {
        &self.errors
    }
}

/// Post-process tokens to handle multi-char operators
/// (rustc_lexer gives single chars, we need ::, ->, etc.)
pub fn cook_tokens(tokens: Vec<Token>) -> Vec<Token> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;
    // Track generic angle bracket depth to avoid combining >> into Shr in generics
    let mut angle_depth: i32 = 0;

    while i < tokens.len() {
        let tok = &tokens[i];

        // Track angle bracket depth for generic contexts
        // Increment on < (but not << or <=), decrement on > (but not >> or >=)
        match &tok.kind {
            TokenKind::Lt => {
                // Check if it's < followed by < or = (which would become << or <=)
                let is_compound = i + 1 < tokens.len() &&
                    tok.span.end == tokens[i + 1].span.start &&
                    matches!(&tokens[i + 1].kind, TokenKind::Lt | TokenKind::Eq);
                if !is_compound {
                    angle_depth += 1;
                }
            }
            TokenKind::Gt => {
                // Check if it's > followed by > or = (which would become >> or >=)
                let is_compound = i + 1 < tokens.len() &&
                    tok.span.end == tokens[i + 1].span.start &&
                    matches!(&tokens[i + 1].kind, TokenKind::Gt | TokenKind::Eq);
                // Only decrement if NOT in a compound AND we're in angle brackets
                if !is_compound && angle_depth > 0 {
                    angle_depth -= 1;
                }
            }
            _ => {}
        }

        // Check for triple tokens first (..=, <<=, >>=)
        if i + 2 < tokens.len() {
            let t1 = &tokens[i];
            let t2 = &tokens[i + 1];
            let t3 = &tokens[i + 2];

            if t1.span.end == t2.span.start && t2.span.end == t3.span.start {
                let combined = match (&t1.kind, &t2.kind, &t3.kind) {
                    (TokenKind::Dot, TokenKind::Dot, TokenKind::Eq) => Some(TokenKind::DotDotEq),
                    (TokenKind::Dot, TokenKind::Dot, TokenKind::Dot) => Some(TokenKind::Ellipsis),
                    (TokenKind::Lt, TokenKind::Lt, TokenKind::Eq) => Some(TokenKind::ShlEq),
                    // Only combine >>= when not inside generics
                    (TokenKind::Gt, TokenKind::Gt, TokenKind::Eq) if angle_depth == 0 => Some(TokenKind::ShrEq),
                    _ => None,
                };

                if let Some(kind) = combined {
                    let span = t1.span.merge(t3.span);
                    result.push(Token::new(kind, span));
                    i += 3;
                    continue;
                }
            }
        }

        // Check for double tokens
        if i + 1 < tokens.len() {
            let tok = &tokens[i];
            let next = &tokens[i + 1];

            if tok.span.end == next.span.start {
                let combined = match (&tok.kind, &next.kind) {
                    (TokenKind::Colon, TokenKind::Colon) => Some(TokenKind::PathSep),
                    (TokenKind::Minus, TokenKind::Gt) => Some(TokenKind::Arrow),
                    (TokenKind::Eq, TokenKind::Gt) => Some(TokenKind::FatArrow),
                    (TokenKind::Eq, TokenKind::Eq) => Some(TokenKind::EqEq),
                    (TokenKind::Not, TokenKind::Eq) => Some(TokenKind::Ne),
                    (TokenKind::Lt, TokenKind::Eq) => Some(TokenKind::Le),
                    (TokenKind::Gt, TokenKind::Eq) => Some(TokenKind::Ge),
                    (TokenKind::And, TokenKind::And) => Some(TokenKind::AndAnd),
                    (TokenKind::Or, TokenKind::Or) => Some(TokenKind::OrOr),
                    (TokenKind::Lt, TokenKind::Lt) => Some(TokenKind::Shl),
                    // Only combine >> when not inside generics
                    (TokenKind::Gt, TokenKind::Gt) if angle_depth == 0 => Some(TokenKind::Shr),
                    (TokenKind::Dot, TokenKind::Dot) => Some(TokenKind::DotDot),
                    (TokenKind::Plus, TokenKind::Eq) => Some(TokenKind::PlusEq),
                    (TokenKind::Minus, TokenKind::Eq) => Some(TokenKind::MinusEq),
                    (TokenKind::Star, TokenKind::Eq) => Some(TokenKind::StarEq),
                    (TokenKind::Slash, TokenKind::Eq) => Some(TokenKind::SlashEq),
                    (TokenKind::Percent, TokenKind::Eq) => Some(TokenKind::PercentEq),
                    (TokenKind::And, TokenKind::Eq) => Some(TokenKind::AndEq),
                    (TokenKind::Or, TokenKind::Eq) => Some(TokenKind::OrEq),
                    (TokenKind::Caret, TokenKind::Eq) => Some(TokenKind::CaretEq),
                    _ => None,
                };

                if let Some(kind) = combined {
                    let span = tok.span.merge(next.span);
                    result.push(Token::new(kind, span));
                    i += 2;
                    continue;
                }
            }
        }

        result.push(tokens[i].clone());
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let (tokens, errors) = Lexer::tokenize("fn main() { }");
        assert!(errors.is_empty());
        assert_eq!(tokens[0].kind, TokenKind::Keyword(Keyword::Fn));
        assert_eq!(tokens[1].kind, TokenKind::Ident("main".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::LParen);
        assert_eq!(tokens[3].kind, TokenKind::RParen);
        assert_eq!(tokens[4].kind, TokenKind::LBrace);
        assert_eq!(tokens[5].kind, TokenKind::RBrace);
    }

    #[test]
    fn test_operators() {
        let (tokens, errors) = Lexer::tokenize("a + b - c * d / e");
        let tokens = cook_tokens(tokens);
        assert!(errors.is_empty());
        assert_eq!(tokens[1].kind, TokenKind::Plus);
        assert_eq!(tokens[3].kind, TokenKind::Minus);
        assert_eq!(tokens[5].kind, TokenKind::Star);
        assert_eq!(tokens[7].kind, TokenKind::Slash);
    }

    #[test]
    fn test_compound_operators() {
        let (tokens, _) = Lexer::tokenize(":: -> => == != <= >=");
        let tokens = cook_tokens(tokens);
        assert_eq!(tokens[0].kind, TokenKind::PathSep);
        assert_eq!(tokens[1].kind, TokenKind::Arrow);
        assert_eq!(tokens[2].kind, TokenKind::FatArrow);
        assert_eq!(tokens[3].kind, TokenKind::EqEq);
        assert_eq!(tokens[4].kind, TokenKind::Ne);
        assert_eq!(tokens[5].kind, TokenKind::Le);
        assert_eq!(tokens[6].kind, TokenKind::Ge);
    }

    #[test]
    fn test_integers() {
        let (tokens, errors) = Lexer::tokenize("42 0xFF 0b1010 1_000");
        assert!(errors.is_empty());
        assert_eq!(tokens[0].kind, TokenKind::Int(42));
        assert_eq!(tokens[1].kind, TokenKind::Int(255));
        assert_eq!(tokens[2].kind, TokenKind::Int(10));
        assert_eq!(tokens[3].kind, TokenKind::Int(1000));
    }

    #[test]
    fn test_floats() {
        let (tokens, errors) = Lexer::tokenize("3.14 1e10 2.5e-3");
        assert!(errors.is_empty());
        assert!(matches!(tokens[0].kind, TokenKind::Float(f) if (f - 3.14).abs() < 0.001));
        assert!(matches!(tokens[1].kind, TokenKind::Float(f) if (f - 1e10).abs() < 1.0));
    }

    #[test]
    fn test_strings() {
        let (tokens, errors) = Lexer::tokenize(r#""hello" "world\n""#);
        assert!(errors.is_empty());
        assert_eq!(tokens[0].kind, TokenKind::Str("hello".to_string()));
        assert_eq!(tokens[1].kind, TokenKind::Str("world\n".to_string()));
    }

    #[test]
    fn test_chars() {
        let (tokens, errors) = Lexer::tokenize("'a' '\\n' '\\x41'");
        assert!(errors.is_empty());
        assert_eq!(tokens[0].kind, TokenKind::Char('a'));
        assert_eq!(tokens[1].kind, TokenKind::Char('\n'));
        assert_eq!(tokens[2].kind, TokenKind::Char('A'));
    }

    #[test]
    fn test_lifetimes() {
        let (tokens, errors) = Lexer::tokenize("'a 'static 'lifetime");
        assert!(errors.is_empty());
        assert_eq!(tokens[0].kind, TokenKind::Lifetime("a".to_string()));
        assert_eq!(tokens[1].kind, TokenKind::Lifetime("static".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Lifetime("lifetime".to_string()));
    }

    #[test]
    fn test_keywords() {
        let (tokens, _) = Lexer::tokenize("fn let if else match struct enum impl trait pub");
        assert_eq!(tokens[0].kind, TokenKind::Keyword(Keyword::Fn));
        assert_eq!(tokens[1].kind, TokenKind::Keyword(Keyword::Let));
        assert_eq!(tokens[2].kind, TokenKind::Keyword(Keyword::If));
        assert_eq!(tokens[3].kind, TokenKind::Keyword(Keyword::Else));
        assert_eq!(tokens[4].kind, TokenKind::Keyword(Keyword::Match));
        assert_eq!(tokens[5].kind, TokenKind::Keyword(Keyword::Struct));
        assert_eq!(tokens[6].kind, TokenKind::Keyword(Keyword::Enum));
        assert_eq!(tokens[7].kind, TokenKind::Keyword(Keyword::Impl));
        assert_eq!(tokens[8].kind, TokenKind::Keyword(Keyword::Trait));
        assert_eq!(tokens[9].kind, TokenKind::Keyword(Keyword::Pub));
    }

    #[test]
    fn test_comments() {
        let (tokens, errors) = Lexer::tokenize("a // comment\nb /* block */ c");
        assert!(errors.is_empty());
        assert_eq!(tokens[0].kind, TokenKind::Ident("a".to_string()));
        assert_eq!(tokens[1].kind, TokenKind::Ident("b".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Ident("c".to_string()));
    }
}
