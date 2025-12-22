// Comprehensive test of bolt features

enum Option {
    None,
    Some(i64),
}

struct Point {
    x: i64,
    y: i64,
}

fn identity<T>(x: T) -> T {
    x
}

fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fn main() -> i64 {
    // Arithmetic
    let a = 10 + 20;

    // Structs
    let p = Point { x: 5, y: 10 };
    let sum = p.x + p.y;

    // Enums and pattern matching
    let opt = Option::Some(100);
    let val = match opt {
        Option::None => 0,
        Option::Some(v) => v,
    };

    // if-let
    let opt2 = Option::Some(50);
    let val2 = if let Option::Some(x) = opt2 {
        x
    } else {
        0
    };

    // Generics
    let g = identity(42);

    // Closures
    let f = |x| x * 2;
    let c = f(10);

    // For loops
    let mut total = 0;
    for i in 0..10 {
        total = total + i;
    }

    // References
    let x = 7;
    let ptr = &x;
    let deref = *ptr;

    // Function calls
    let fib = fibonacci(10);

    // Result: 30 + 15 + 100 + 50 + 42 + 20 + 45 + 7 + 55 = 364
    a + sum + val + val2 + g + c + total + deref + fib
}
