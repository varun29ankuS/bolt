// Test static constructor inside another constructor

struct Inner {
    value: i64,
}

impl Inner {
    fn new(v: i64) -> Inner {
        Inner { value: v }
    }

    fn get(&self) -> i64 {
        self.value
    }
}

struct Outer {
    data: i64,
    inner: Inner,
}

impl Outer {
    fn new(d: i64, v: i64) -> Outer {
        Outer {
            data: d,
            inner: Inner::new(v),  // Static method call inside constructor
        }
    }

    fn get_inner(&self) -> i64 {
        self.inner.get()
    }
}

fn main() {
    let o = Outer::new(10, 200);
    print(o.data);        // Should print 10
    print(o.get_inner()); // Should print 200
    print(42);            // Success marker
}
