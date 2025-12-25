// Test Vec<T> basic operations

fn main() {
    let mut v: Vec<i32> = Vec::new();

    // Push elements
    v.push(10);
    v.push(20);
    v.push(30);

    // Length
    print(v.len());  // 3

    // Indexing
    print(v[0]);  // 10
    print(v[1]);  // 20
    print(v[2]);  // 30

    // Pop
    let popped = v.pop();
    print(popped);  // 30
    print(v.len()); // 2

    // Modify via index
    v[0] = 100;
    print(v[0]);  // 100
}
