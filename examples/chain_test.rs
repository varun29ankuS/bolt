// Test struct field access chain

struct Inner {
    value: i64,
}

struct Outer {
    data: i64,
    inner: Inner,
}

fn main() {
    // Direct construction and access
    let o = Outer {
        data: 10,
        inner: Inner { value: 200 }
    };

    // Test each step
    print(o.data);  // Step 1: scalar field - should print 10

    // Get inner as separate variable
    let inner_ref = o.inner;
    print(42);  // Marker
}
