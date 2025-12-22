fn first<T, U>(a: T, b: U) -> T {
    a
}

fn second<T, U>(a: T, b: U) -> U {
    b
}

fn main() -> i64 {
    first(10, 20) + second(30, 40)
}
