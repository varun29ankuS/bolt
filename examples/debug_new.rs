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
    print(a.value);  // should be 10
}
