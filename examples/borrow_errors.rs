// Test that borrow checker catches actual errors

fn main() {
    // Test 1: Use after move
    let s = String::from("hello");
    let t = s;  // move
    println!("{}", s);  // ERROR: use after move

    // Test 2: Double mutable borrow
    let mut v = Vec::new();
    let r1 = &mut v;
    let r2 = &mut v;  // ERROR: second mutable borrow
    r1.push(1);
    r2.push(2);

    // Test 3: Mutable while immutably borrowed
    let mut x = 42;
    let r = &x;
    let m = &mut x;  // ERROR: mutable borrow while immutably borrowed
    println!("{} {}", r, m);
}
