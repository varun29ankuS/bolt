fn main() -> i64 {
    let a = 10;
    let b = 20;
    let ptr_a = &a;
    let ptr_b = &b;
    *ptr_a + *ptr_b
}
