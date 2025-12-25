// Trait bound test

trait Printable {
    fn value(&self) -> i32;
}

trait Addable {
    fn add(&self, n: i32) -> i32;
}

struct Num {
    n: i32,
}

impl Printable for Num {
    fn value(&self) -> i32 {
        self.n
    }
}

impl Addable for Num {
    fn add(&self, x: i32) -> i32 {
        self.n + x
    }
}

fn print_it<T: Printable>(x: T) {
    print(x.value());
}

fn add_ten<T: Addable>(x: T) -> i32 {
    x.add(10)
}

fn main() {
    let n = Num { n: 42 };
    print_it(n);  // 42

    let m = Num { n: 5 };
    print(add_ten(m));  // 15
}
