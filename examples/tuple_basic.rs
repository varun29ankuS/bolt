// Basic tuple test

fn main() {
    // Simple pair
    let pair = (10, 20);
    print(pair.0);  // 10
    print(pair.1);  // 20

    // Triple
    let triple = (1, 2, 3);
    print(triple.0 + triple.1 + triple.2);  // 6

    // Nested access
    let sum = pair.0 + pair.1;
    print(sum);  // 30
}
