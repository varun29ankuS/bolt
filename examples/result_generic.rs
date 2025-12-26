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
}

fn main() {
    let x: Result<i32, i32> = Result::Ok(42);
    if x.is_ok() {
        print(1);  // Should print 1
    } else {
        print(0);
    }

    let y: Result<i32, i32> = Result::Err(99);
    if y.is_err() {
        print(2);  // Should print 2
    } else {
        print(0);
    }
}
