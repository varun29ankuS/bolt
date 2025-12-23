// Test if o.inner returns an address or a value

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

    // Direct access to Outer field - should print 10
    print(o.data);

    // Access Inner through Outer - should print 200
    // If it prints a large number, we're getting an address
    let x = o.inner.value;
    print(x);
}
