// Test string literals and print functions

fn main() -> i64 {
    // Test print functions
    bolt_print_int(42);

    // String literal - returns pointer to stack-allocated bytes
    let s = "Hello";
    bolt_print_str(s, 5);  // Print "Hello"

    // Test simple string length calculation
    // For now, just return a value to verify execution
    42
}
