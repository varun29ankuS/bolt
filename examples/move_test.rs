// Test move detection

fn main() {
    let s: String = String::from("hello");
    let t = s;  // move
    println!("{}", s);  // ERROR: use after move
}
