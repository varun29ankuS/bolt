// Test slice types (&[T])

fn sum_slice(s: &[i64]) -> i64 {
    let mut total = 0;
    let mut i = 0;
    while i < s.len() {
        total = total + s[i];
        i = i + 1;
    }
    total
}

fn first_elem(s: &[i64]) -> i64 {
    s[0]
}

fn main() {
    let arr = [10, 20, 30];

    // Pass array as slice
    let total = sum_slice(&arr);
    print(total);  // 60

    let first = first_elem(&arr);
    print(first);  // 10

    // Slice length
    print(arr.len());  // 3
}
