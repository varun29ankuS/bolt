// Simplest nested struct test

struct Inner {
    value: i64,
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
    print(o.inner.value);  // Should print 200
}
