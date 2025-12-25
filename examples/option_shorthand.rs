// Test shorthand Some(x) / None syntax

enum Option {
    Some(i32),
    None,
}

fn main() {
    let opt = Some(42);
    
    match opt {
        Some(x) => print(x),
        None => print(0),
    }
    
    if let Some(y) = opt {
        print(y);
    }
}
