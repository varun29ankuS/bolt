fn main() {
    let mut v: Vec<i64> = Vec::new();
    v.push(10);
    v.push(20);

    print(v.len());  // 2

    // This should panic with "index out of bounds"
    let bad = v[5];
    print(bad);  // Should not reach here
}
