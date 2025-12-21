fn count_down(n: i64) -> i64 {
    if n <= 0 {
        0
    } else {
        n + count_down(n - 1)
    }
}

fn main() -> i64 {
    count_down(10)
}
