// Test: Generic Option<T> with methods

enum Option<T> {
    None,
    Some(T),
}

impl<T> Option<T> {
    fn is_some(&self) -> bool {
        match self {
            Option::Some(_) => true,
            Option::None => false,
        }
    }

    fn is_none(&self) -> bool {
        match self {
            Option::Some(_) => false,
            Option::None => true,
        }
    }
}

fn main() {
    let x: Option<i32> = Option::Some(42);
    let y: Option<i32> = Option::None;

    if x.is_some() {
        print(1);  // Should print 1
    }

    if y.is_none() {
        print(2);  // Should print 2
    }
}
