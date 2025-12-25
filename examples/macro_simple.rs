// Simplest possible macro test

macro_rules! answer {
    () => {
        42
    };
}

fn main() {
    let x = answer!();
    print(x);
}
