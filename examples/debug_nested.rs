// Debug nested struct issue

struct Inner {
    value: i64,
}

struct Outer {
    data: i64,
    inner: Inner,
}

fn main() {
    let i = Inner { value: 200 };
    print(i.value);  // Should print 200 - test simple struct field

    let o = Outer {
        data: 10,
        inner: Inner { value: 300 }
    };
    print(o.data);  // Should print 10
}
