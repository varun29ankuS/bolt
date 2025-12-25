// Test explicit lifetime annotations

struct Wrapper<'a> {
    data: &'a i64,
}

impl<'a> Wrapper<'a> {
    fn new(data: &'a i64) -> Wrapper<'a> {
        Wrapper { data }
    }

    fn get(&self) -> i64 {
        *self.data
    }
}

fn identity<'a>(x: &'a i64) -> &'a i64 {
    x
}

fn main() {
    let x = 42;
    let r = identity(&x);
    print(*r);  // 42

    let y = 100;
    let w = Wrapper::new(&y);
    print(w.get());  // 100
}
