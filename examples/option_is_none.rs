enum Option {
    Some(i32),
    None,
}

fn main() {
    let opt = None;
    if opt.is_none() {
        print(1);
    }
}
