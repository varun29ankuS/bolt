// Call two methods that return primitives

struct Counter {
    value: i32,
}

impl Counter {
    fn get(&self) -> i32 {
        self.value
    }
}

fn main() {
    let c = Counter { value: 10 };
    print(c.get());  // 10
    print(c.get());  // 10
}
