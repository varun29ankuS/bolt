struct Counter {
    value: i32,
}

fn main() {
    let c = Counter { value: 10 };
    print(c.value);  // 10
    print(c.value);  // 10
}
