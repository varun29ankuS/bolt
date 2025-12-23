// Generic impl test

struct Wrapper<T> {
    value: T,
}

impl<T> Wrapper<T> {
    fn new(value: T) -> Wrapper<T> {
        Wrapper { value }
    }
}

fn main() {
    let w = Wrapper::new(42);
    print(w.value);
}
