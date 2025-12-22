// Test Result type and ? operator

enum Result {
    Ok(i64),
    Err(i64),
}

fn divide(a: i64, b: i64) -> Result {
    if b == 0 {
        Result::Err(1)  // Error code for division by zero
    } else {
        Result::Ok(a / b)
    }
}

fn compute(x: i64) -> Result {
    // Use ? to propagate errors
    let a = divide(100, x)?;  // If Err, return early
    let b = divide(a, 2)?;    // If Err, return early
    Result::Ok(b)
}

fn main() -> i64 {
    // Test successful case
    let result1 = compute(5);  // 100/5=20, 20/2=10 -> Ok(10)
    let val1 = match result1 {
        Result::Ok(v) => v,
        Result::Err(_) => 0,
    };

    // Test error case (division by zero)
    let result2 = compute(0);  // divide by 0 -> Err(1)
    let val2 = match result2 {
        Result::Ok(_) => 0,
        Result::Err(e) => e,
    };

    val1 + val2 * 100  // 10 + 1*100 = 110
}
