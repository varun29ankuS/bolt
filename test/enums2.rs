enum Option {
    None,
    Some(i64),
}

fn main() -> i64 {
    let x = Option::None;
    match x {
        Option::None => 99,
        Option::Some(val) => val,
    }
}
