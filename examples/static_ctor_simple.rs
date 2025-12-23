// Simplest static constructor test

struct Inner {
    value: i64,
}

impl Inner {
    fn new(v: i64) -> Inner {
        Inner { value: v }
    }
}

struct Outer {
    data: i64,
    inner: Inner,
}

fn main() {
    // Test just the static constructor
    let i = Inner::new(42);
    print(i.value);  // Should print 42

    // Test nested struct with static constructor
    let o = Outer {
        data: 10,
        inner: Inner::new(200),
    };
    print(o.data);        // Should print 10
    print(o.inner.value); // Should print 200
}
