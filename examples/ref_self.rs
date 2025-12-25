// Test &self methods - simplified

struct Counter {
    value: i32,
}

impl Counter {
    fn new(v: i32) -> Counter {
        Counter { value: v }
    }

    fn get(&self) -> i32 {
        self.value
    }
}

fn main() {
    let c = Counter::new(10);
    print(c.get());
}
