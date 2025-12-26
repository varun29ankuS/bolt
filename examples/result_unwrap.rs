enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> Result<T, E> {
    fn unwrap_or(self, default: T) -> T {
        match self {
            Result::Ok(val) => val,
            Result::Err(_) => default,
        }
    }
}

fn main() {
    let x: Result<i32, i32> = Result::Ok(42);
    print(x.unwrap_or(100));  // Should print 42

    let y: Result<i32, i32> = Result::Err(99);
    print(y.unwrap_or(100));  // Should print 100
}
