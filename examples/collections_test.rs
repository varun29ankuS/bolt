// Test comprehensive Vec and String functionality

fn main() {
    // Vec tests
    let mut numbers: Vec<i64> = Vec::new();
    numbers.push(1);
    numbers.push(2);
    numbers.push(3);
    numbers.push(4);
    numbers.push(5);

    print(numbers.len());      // 5
    print(numbers.sum());      // 15

    // Test get
    print(numbers.get(0));     // 1
    print(numbers.get(4));     // 5

    // Vec::with_capacity
    let mut large: Vec<i64> = Vec::with_capacity(100);
    print(large.capacity());   // 100
    print(large.len());        // 0

    large.push(42);
    print(large.len());        // 1
    print(large.get(0));       // 42

    // String tests
    let greeting = String::from("Hello");
    print(greeting.len());     // 5

    let mut name = String::new();
    print(name.len());         // 0

    // Compute fibonacci and store in vec
    let mut fib: Vec<i64> = Vec::new();
    fib.push(1);
    fib.push(1);

    let a = fib.get(0);
    let b = fib.get(1);
    fib.push(a + b);  // 2

    let a = fib.get(1);
    let b = fib.get(2);
    fib.push(a + b);  // 3

    let a = fib.get(2);
    let b = fib.get(3);
    fib.push(a + b);  // 5

    print(fib.len());    // 5
    print(fib.sum());    // 12 (1+1+2+3+5)
}
