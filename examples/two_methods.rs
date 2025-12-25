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
    print(Counter::new(10).get());  // first method call
    print(Counter::new(20).get());  // second method call
}
