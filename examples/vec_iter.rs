// Vec iterator test

fn main() {
    let v: Vec<i64> = Vec::new();
    v.push(10);
    v.push(20);
    v.push(30);

    let mut sum = 0;
    for x in v.iter() {
        sum = sum + x;
    }
    print(sum);  // should be 60
}
