// Test impl Trait return types

trait Doubler {
    fn double(&self) -> i64;
}

struct Number {
    value: i64,
}

impl Doubler for Number {
    fn double(&self) -> i64 {
        self.value * 2
    }
}

// Function returning impl Trait
fn make_doubler(val: i64) -> impl Doubler {
    Number { value: val }
}

fn main() {
    let d = make_doubler(21);
    print(d.double());  // Should print 42

    let d2 = make_doubler(50);
    print(d2.double()); // Should print 100
}
