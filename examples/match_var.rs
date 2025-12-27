fn test(x: i32, default: i32) -> i32 {
    if x > 0 {
        x
    } else {
        default
    }
}

fn main() {
    print(test(5, 99));   // Should print 5
    print(test(0, 99));   // Should print 99
}
