// Test if let Some(x) pattern

enum Option {
    Some(i32),
    None,
}

fn main() {
    let opt = Option::Some(42);
    
    if let Option::Some(x) = opt {
        print(x);
    } else {
        print(0);
    }
    
    let opt2 = Option::None;
    if let Option::Some(y) = opt2 {
        print(y);
    } else {
        print(99);
    }
}
