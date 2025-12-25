// Test &mut self method calls

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

    fn increment(&mut self) {
        self.value = self.value + 1;
    }

    fn add(&mut self, n: i32) {
        self.value = self.value + n;
    }
}

fn main() {
    let mut c = Counter::new(10);
    print(c.get());  // 10
    c.increment();
    print(c.get());  // 11
    c.add(5);
    print(c.get());  // 16
}
