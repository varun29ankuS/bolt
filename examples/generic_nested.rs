// MyBox<T>-like pattern

struct MyBox<T> {
    value: T,
}

impl<T> MyBox<T> {
    fn create(value: T) -> MyBox<T> {
        MyBox { value }
    }

    fn get(self) -> T {
        self.value
    }
}

struct Holder<T> {
    inner: MyBox<T>,
}

impl<T> Holder<T> {
    fn wrap(value: T) -> Holder<T> {
        Holder { inner: MyBox::create(value) }
    }

    fn unwrap(self) -> T {
        self.inner.get()
    }
}

fn main() {
    let b = MyBox::create(42);
    print(b.get());

    let b2 = MyBox::create(100);
    print(b2.value);

    // Nested generic usage
    let h = Holder::wrap(200);
    print(h.unwrap());
}
