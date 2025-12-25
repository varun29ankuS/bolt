// Simplest &mut self test

struct Cell {
    value: i32,
}

impl Cell {
    fn new(v: i32) -> Cell {
        Cell { value: v }
    }

    fn set(&mut self, v: i32) {
        self.value = v;
    }

    fn get(&self) -> i32 {
        self.value
    }
}

fn main() {
    let mut c = Cell::new(10);
    print(c.get());  // 10
    c.set(42);
    print(c.get());  // should be 42
}
