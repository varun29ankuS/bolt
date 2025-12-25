// Test Option.unwrap() method

enum Option {
    Some(i32),
    None,
}

fn main() {
    let opt = Some(42);
    let x = opt.unwrap();
    print(x);  // 42
    
    let opt2 = Some(100);
    print(opt2.unwrap());  // 100
    
    // Test is_some/is_none
    let opt3 = Some(1);
    let opt4 = None;
    
    if opt3.is_some() {
        print(1);  // should print
    }
    if opt4.is_none() {
        print(2);  // should print
    }
}
