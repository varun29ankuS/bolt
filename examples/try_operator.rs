// Test ? operator with Result enum

enum Result {
    Ok(i32),
    Err(i32),
}

fn maybe_fail(x: i32) -> Result {
    if x > 0 {
        Result::Ok(x * 2)
    } else {
        Result::Err(-1)
    }
}

fn process(x: i32) -> Result {
    let val = maybe_fail(x)?;
    Result::Ok(val + 10)
}

fn main() {
    // Test success case
    let r1 = process(5);
    match r1 {
        Result::Ok(v) => print(v),   // 5*2 + 10 = 20
        Result::Err(e) => print(e),
    }

    // Test error case
    let r2 = process(-1);
    match r2 {
        Result::Ok(v) => print(v),
        Result::Err(e) => print(e),  // -1 (error propagated)
    }
}
