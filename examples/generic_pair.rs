// Generic struct with two type parameters

struct Pair<T, U> {
    first: T,
    second: U,
}

impl<T, U> Pair<T, U> {
    fn new(first: T, second: U) -> Pair<T, U> {
        Pair { first, second }
    }

    fn get_first(self) -> T {
        self.first
    }

    fn get_second(self) -> U {
        self.second
    }
}

fn main() {
    let p = Pair::new(10, 20);
    print(p.first);
    print(p.second);

    let p2 = Pair::new(100, 200);
    print(p2.get_first());

    let p3 = Pair::new(300, 400);
    print(p3.get_second());
}
