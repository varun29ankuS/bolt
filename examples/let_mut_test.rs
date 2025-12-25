// Test let mut with struct

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
    let mut c = Cell::new(10);  // This is the difference - let mut vs let
    print(c.get());  // should be 10
}
