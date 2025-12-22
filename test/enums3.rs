enum Result {
    Ok(i64),
    Err(i64),
}

fn main() -> i64 {
    let r1 = Result::Ok(100);
    let r2 = Result::Err(50);

    let v1 = match r1 {
        Result::Ok(x) => x,
        Result::Err(e) => e,
    };

    let v2 = match r2 {
        Result::Ok(x) => x,
        Result::Err(e) => e,
    };

    v1 + v2
}
