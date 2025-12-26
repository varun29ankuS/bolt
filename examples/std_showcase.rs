// Showcase: Pure Bolt-compiled std types
// Option<T> and Result<T, E> implemented as real Rust code

enum Option<T> {
    None,
    Some(T),
}

impl<T> Option<T> {
    fn is_some(self) -> bool {
        match self {
            Option::Some(_) => true,
            Option::None => false,
        }
    }

    fn is_none(self) -> bool {
        match self {
            Option::Some(_) => false,
            Option::None => true,
        }
    }

    fn unwrap_or(self, default: T) -> T {
        match self {
            Option::Some(val) => val,
            Option::None => default,
        }
    }
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> Result<T, E> {
    fn is_ok(self) -> bool {
        match self {
            Result::Ok(_) => true,
            Result::Err(_) => false,
        }
    }

    fn is_err(self) -> bool {
        match self {
            Result::Ok(_) => false,
            Result::Err(_) => true,
        }
    }

    fn unwrap_or(self, default: T) -> T {
        match self {
            Result::Ok(val) => val,
            Result::Err(_) => default,
        }
    }
}

// Helper function demonstrating division with Result
fn divide(a: i32, b: i32) -> Result<i32, i32> {
    if b == 0 {
        Result::Err(-1)
    } else {
        Result::Ok(a / b)
    }
}

fn main() {
    print(111);  // Marker: starting tests

    // Test Option<T>
    let some_val: Option<i32> = Option::Some(42);
    let none_val: Option<i32> = Option::None;

    print(some_val.unwrap_or(0));   // 42
    print(none_val.unwrap_or(99));  // 99

    // Test Result<T, E>
    let ok_val: Result<i32, i32> = Result::Ok(100);
    let err_val: Result<i32, i32> = Result::Err(0);

    print(ok_val.unwrap_or(0));    // 100
    print(err_val.unwrap_or(200)); // 200

    // Test divide function
    let r1 = divide(20, 4);
    print(r1.unwrap_or(0));  // 5

    let r2 = divide(10, 0);
    if r2.is_err() {
        print(999);  // 999 - division by zero detected
    }

    print(222);  // Marker: tests complete
}
