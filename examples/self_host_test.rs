// Self-hosting smoke test - simplified

struct Span {
    start: i64,
    end: i64,
}

impl Span {
    fn new(start: i64, end: i64) -> Span {
        Span { start, end }
    }

    fn len(&self) -> i64 {
        self.end - self.start
    }
}

struct Token {
    kind: i64,
    span: Span,
}

impl Token {
    fn new(kind: i64, start: i64, end: i64) -> Token {
        Token {
            kind,
            span: Span::new(start, end),
        }
    }

    fn span_len(&self) -> i64 {
        self.span.len()
    }
}

fn main() {
    // Test Span
    let s = Span::new(10, 20);
    print(s.len());  // Should print 10

    // Test Token with nested Span
    let t = Token::new(1, 5, 15);
    print(t.span_len());  // Should print 10

    // Test Vec
    let mut tokens: Vec<i64> = Vec::new();
    tokens.push(1);
    tokens.push(2);
    tokens.push(3);
    print(tokens.len());  // Should print 3

    print(42);  // Success marker
}
