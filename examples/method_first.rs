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
    print(Counter::new(10).get());  // method call first
    print(42);  // then constant
}
