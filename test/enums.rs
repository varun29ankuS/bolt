enum Option {
    None,
    Some(i64),
}

fn main() -> i64 {
    let x = Option::Some(42);
    match x {
        Option::None => 0,
        Option::Some(val) => val,
    }
}
