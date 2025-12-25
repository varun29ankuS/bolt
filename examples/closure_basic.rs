// Basic closure test

fn main() {
    // Simple closure
    let double = |x: i32| x * 2;
    print(double(5));  // 10

    // Two-argument closure
    let add = |a: i32, b: i32| a + b;
    print(add(10, 20));  // 30

    // Closure capturing outer variable
    let base = 100;
    let add_base = |n: i32| n + base;
    print(add_base(5));  // 105
}
