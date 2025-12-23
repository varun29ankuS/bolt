// Simplest struct test

struct Counter {
    value: i64,
}

impl Counter {
    fn get(&self) -> i64 {
        self.value
    }
}

fn main() {
    let c = Counter { value: 42 };
    let v = c.get();
    print(v);
}
