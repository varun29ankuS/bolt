// Test method self parameters with Chumsky parser

struct Counter {
    value: i32,
}

impl Counter {
    fn new() -> Counter {
        Counter { value: 0 }
    }

    fn get(&self) -> i32 {
        self.value
    }

    fn set(&mut self, v: i32) {
        self.value = v;
    }

    fn increment(&mut self) {
        self.value = self.value + 1;
    }

    fn into_value(self) -> i32 {
        self.value
    }
}

fn main() {
    let mut c = Counter::new();
    c.set(10);
    c.increment();
    print(c.get());  // Should print 11
    print(c.into_value());  // Should print 11
    print(999);  // Success marker
}
