// Test floating point operations with proper types

fn main() -> i64 {
    // Float arithmetic
    let a: f64 = 3.5;
    let b: f64 = 2.5;

    let sum = a + b;      // 6.0
    let diff = a - b;     // 1.0
    let prod = a * b;     // 8.75
    let quot = a / b;     // 1.4

    // Float comparisons
    let lt = if a < b { 1 } else { 0 };   // 0
    let gt = if a > b { 1 } else { 0 };   // 1
    let eq = if a == a { 1 } else { 0 };  // 1

    // Mixed - return integer result
    // sum=6, diff=1, prod=8.75, quot=1.4
    // We can only return i64, so just return comparison results
    lt + gt + eq  // 0 + 1 + 1 = 2
}
