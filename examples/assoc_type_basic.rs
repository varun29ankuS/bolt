// Test associated types in traits

trait Container {
    type Item;

    fn get(&self) -> Self::Item;
}

struct IntBox {
    value: i64,
}

impl Container for IntBox {
    type Item = i64;

    fn get(&self) -> i64 {
        self.value
    }
}

struct PairBox {
    first: i64,
    second: i64,
}

impl Container for PairBox {
    type Item = i64;

    fn get(&self) -> i64 {
        self.first + self.second
    }
}

fn main() {
    let box1 = IntBox { value: 42 };
    print(box1.get());  // 42

    let box2 = PairBox { first: 10, second: 20 };
    print(box2.get());  // 30
}
