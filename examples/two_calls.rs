// Test two method calls

struct Cell {
    value: i32,
}

impl Cell {
    fn new(v: i32) -> Cell {
        Cell { value: v }
    }

    fn get(&self) -> i32 {
        self.value
    }
}

fn main() {
    let c = Cell::new(10);
    print(c.get());  // 10
    print(c.get());  // should also be 10
}
