// Generic struct with methods

struct Container<T> {
    value: T,
}

impl<T> Container<T> {
    fn new(value: T) -> Container<T> {
        Container { value }
    }

    fn get(self) -> T {
        self.value
    }
}

fn main() {
    let c = Container::new(42);
    let v = c.get();
    print(v);
}
