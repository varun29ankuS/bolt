// Test: Generic enum with methods - production quality

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

    fn unwrap(self) -> T {
        match self {
            Option::Some(val) => val,
            Option::None => panic("called unwrap on None"),
        }
    }

    fn unwrap_or(self, default: T) -> T {
        match self {
            Option::Some(val) => val,
            Option::None => default,
        }
    }

    fn map<U, F>(self, f: F) -> Option<U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Option::Some(val) => Option::Some(f(val)),
            Option::None => Option::None,
        }
    }

    fn and_then<U, F>(self, f: F) -> Option<U>
    where
        F: FnOnce(T) -> Option<U>,
    {
        match self {
            Option::Some(val) => f(val),
            Option::None => Option::None,
        }
    }
}

fn main() {
    let some_val: Option<i32> = Option::Some(42);
    let none_val: Option<i32> = Option::None;

    // Test is_some / is_none
    if some_val.is_some() {
        print(1);  // Should print 1
    }

    if none_val.is_none() {
        print(2);  // Should print 2
    }

    // Test unwrap
    let x = Option::Some(99);
    print(x.unwrap());  // Should print 99

    // Test unwrap_or
    let y: Option<i32> = Option::None;
    print(y.unwrap_or(100));  // Should print 100
}
