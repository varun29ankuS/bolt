fn main() {
    // Test v[i] syntax
    let mut v: Vec<i64> = Vec::new();
    v.push(100);
    v.push(200);
    v.push(300);

    // Read using index syntax (should use native indexing with bounds check)
    let first = v[0];
    let second = v[1];
    let third = v[2];

    print(first);   // 100
    print(second);  // 200
    print(third);   // 300

    // Write using index syntax
    v[1] = 999;
    print(v[1]);    // 999

    // Verify other elements unchanged
    print(v[0]);    // 100
    print(v[2]);    // 300

    // Test sum still works
    print(v.sum()); // 100 + 999 + 300 = 1399
}
