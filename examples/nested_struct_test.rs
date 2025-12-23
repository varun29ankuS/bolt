// Test nested struct access

struct Inner {
    value: i64,
}

impl Inner {
    fn get(&self) -> i64 {
        self.value
    }
}

struct Outer {
    inner: Inner,
}

impl Outer {
    fn get_inner_value(&self) -> i64 {
        self.inner.value
    }
}

fn main() {
    let inner = Inner { value: 100 };
    print(inner.get());  // Should print 100

    let outer = Outer { inner: Inner { value: 200 } };
    print(outer.inner.value);  // Direct field access - should print 200

    print(outer.get_inner_value());  // Method call - should print 200
}
