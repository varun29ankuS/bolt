// Test method call on nested struct field

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
        self.inner.get()  // This is the problematic pattern
    }
}

fn main() {
    let o = Outer {
        inner: Inner { value: 123 }
    };

    print(o.inner.get());       // Direct chain
    print(o.get_inner_value()); // Through method
}
