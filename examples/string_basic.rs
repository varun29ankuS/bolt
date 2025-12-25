// Test String basic operations

fn main() {
    // Create from literal
    let s = String::from("Hello");
    print(s.len());  // 5

    // Create empty and push
    let mut s2 = String::new();
    print(s2.len());  // 0
    s2.push_str("World");
    print(s2.len());  // 5

    // Print the string
    let greeting = String::from("Hello, Bolt!");
    print(greeting);
}
