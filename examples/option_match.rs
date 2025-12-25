// Test Option<T> pattern matching

enum Option {
    Some(i32),
    None,
}

fn main() {
    let opt = Option::Some(42);
    
    match opt {
        Option::Some(x) => print(x),
        Option::None => print(0),
    }
}
