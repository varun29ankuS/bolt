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
    data: i64,
    inner: Inner,
}

fn main() {
    let o = Outer {
        data: 10,
        inner: Inner { value: 200 }
    };

    print(o.data);         // Should print 10
    print(o.inner.value);  // Should print 200 (direct access)

    // This is the problematic call:
    let v = o.inner.get();  // Method call on nested field
    print(v);               // Should print 200
}
