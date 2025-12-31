// Test borrow checker behavior

fn main() {
    // Test 1: Use after move
    let s = String::from("hello");
    let t = s;  // move
    // println!("{}", s);  // ERROR: use after move - should be caught

    // Test 2: Double mutable borrow
    let mut v = Vec::new();
    let r1 = &mut v;
    // let r2 = &mut v;  // ERROR: second mutable borrow - should be caught
    r1.push(1);

    // Test 3: Mutable and immutable borrow conflict
    let mut x = 42;
    let r = &x;
    // let m = &mut x;  // ERROR: mutable borrow while immutably borrowed
    println!("{}", r);

    // Test 4: Copy types don't move
    let n: i32 = 10;
    let m = n;  // copy, not move
    println!("{} {}", n, m);  // OK: both are valid
}
