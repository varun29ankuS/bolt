// Array iterator test - using direct array literal

fn main() {
    let mut sum = 0;
    // Use array literal directly in the for loop
    for x in [1, 2, 3, 4, 5].iter() {
        sum = sum + x;
    }
    print(sum);  // should be 15
}
