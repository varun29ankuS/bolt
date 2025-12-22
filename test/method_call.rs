// Test method resolution via impl blocks

struct Counter {
    value: i64,
}

impl Counter {
    fn new() -> Counter {
        Counter { value: 0 }
    }

    fn increment(&mut self) -> i64 {
        self.value = self.value + 1;
        self.value
    }

    fn get(&self) -> i64 {
        self.value
    }
}

fn main() -> i64 {
    let mut c = Counter::new();
    c.increment();
    c.increment();
    c.get()
}
