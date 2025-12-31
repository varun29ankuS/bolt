// Simple move detection test (no macros)

fn take_string(s: String) {
    // consumes s
}

fn main() {
    let s: String = String::from("hello");
    let t = s;  // move s to t
    take_string(t);  // OK - t is valid
    let x = s;  // ERROR: use of moved value s
}
