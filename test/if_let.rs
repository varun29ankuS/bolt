enum Option {
    None,
    Some(i64),
}

fn main() -> i64 {
    let x = Option::Some(42);
    if let Option::Some(val) = x {
        val
    } else {
        0
    }
}
