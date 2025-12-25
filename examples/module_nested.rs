// Nested module test

mod outer {
    pub mod inner {
        pub fn value() -> i64 {
            42
        }
    }

    pub fn get_inner() -> i64 {
        inner::value()
    }
}

fn main() {
    print(outer::inner::value());  // 42
    print(outer::get_inner());     // should be 42
}
