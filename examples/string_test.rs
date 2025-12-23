fn main() {
    // Test String::new()
    let mut s = String::new();

    // Test String::from()
    let greeting = String::from("Hello");
    let len = greeting.len();
    print(len);  // Should print 5
}
