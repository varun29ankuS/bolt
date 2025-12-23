// Simplest possible nested test

struct A {
    x: i64,
}

struct B {
    a: A,
}

fn main() {
    // Test 1: Direct struct field
    let a = A { x: 42 };
    print(a.x);  // Should print 42

    // Test 2: Nested struct field
    let b = B { a: A { x: 99 } };
    print(b.a.x);  // Should print 99
}
