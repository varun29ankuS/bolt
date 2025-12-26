// Minimal test - what works?

enum MyOption {
    None,
    Some(i32),
}

fn main() {
    let x = MyOption::Some(42);
    
    match x {
        MyOption::Some(val) => print(val),
        MyOption::None => print(0),
    }
}
