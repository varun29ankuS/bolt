// Equivalent to benchmark.rs for cargo comparison
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fn factorial(n: i64) -> i64 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn main() {
    let fib = fibonacci(10);
    let fact = factorial(6);
    println!("{}", fib + fact);  // 55 + 720 = 775... wait that's different
}
