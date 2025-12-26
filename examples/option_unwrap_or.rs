enum Option<T> {
    None,
    Some(T),
}

impl<T> Option<T> {
    fn unwrap_or(self, default: T) -> T {
        match self {
            Option::Some(val) => val,
            Option::None => default,
        }
    }
}

fn main() {
    let x: Option<i32> = Option::Some(42);
    print(x.unwrap_or(100));  // Should print 42
    
    let y: Option<i32> = Option::None;
    print(y.unwrap_or(99));  // Should print 99
}
