// Test file with intentional borrow errors
// This should trigger warnings in async mode

fn main() -> i64 {
    let mut x = 42;
    let y = &mut x;  // mutable borrow
    let z = &x;      // immutable borrow while mutably borrowed - ERROR
    *y + *z
}
