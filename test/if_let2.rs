enum Option {
    None,
    Some(i64),
}

fn main() -> i64 {
    let x = Option::None;
    if let Option::Some(val) = x {
        val
    } else {
        99
    }
}
