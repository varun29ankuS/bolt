// Float basic operations test

fn main() {
    let x: f64 = 3.14;
    let y: f64 = 2.0;

    // Print floats - these should NOT trigger "use of moved value"
    print(x);       // 3.14
    print(y);       // 2.0

    // Use x and y multiple times - Copy types should allow this
    let sum: f64 = x + y;
    print(sum);     // 5.14

    let diff: f64 = x - y;
    print(diff);    // 1.14

    let prod: f64 = x * y;
    print(prod);    // 6.28

    let quot: f64 = x / y;
    print(quot);    // 1.57
}
