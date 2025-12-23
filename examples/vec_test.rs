fn main() {
    // Test Vec::new()
    let mut v: Vec<i64> = Vec::new();

    // Test push
    v.push(10);
    v.push(20);
    v.push(30);

    // Test len
    let length = v.len();
    print(length);  // Should print 3

    // Test sum
    let total = v.sum();
    print(total);  // Should print 60

    // Test get
    let first = v.get(0);
    print(first);  // Should print 10

    let second = v.get(1);
    print(second);  // Should print 20
}
