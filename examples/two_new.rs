// Call Counter::new twice

struct Counter {
    value: i32,
}

impl Counter {
    fn new(v: i32) -> Counter {
        Counter { value: v }
    }
}

fn main() {
    let a = Counter::new(10);
    let b = Counter::new(20);
    print(a.value);  // 10
    print(b.value);  // 20
}
