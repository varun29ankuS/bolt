// Simple dyn Trait test - just type checking for now

trait Adder {
    fn add_ten(&self) -> i64;
}

struct Number {
    value: i64,
}

impl Adder for Number {
    fn add_ten(&self) -> i64 {
        self.value + 10
    }
}

fn call_adder(a: &dyn Adder) -> i64 {
    a.add_ten()
}

fn main() {
    let n = Number { value: 32 };
    // For now, dyn Trait acts like impl Trait - static dispatch
    print(call_adder(&n));  // Should print 42
}
