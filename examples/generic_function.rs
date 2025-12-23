// Generic function tests

fn identity<T>(x: T) -> T {
    x
}

fn first<T, U>(a: T, b: U) -> T {
    a
}

fn main() {
    let a = identity(42);
    let b = identity(100);
    let c = first(10, 20);
    print(a);
    print(b);
    print(c);
}
