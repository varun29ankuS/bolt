// Test basic macro_rules! functionality

macro_rules! double {
    ($x:expr) => {
        $x + $x
    };
}

macro_rules! add_one {
    ($x:expr) => {
        $x + 1
    };
}

fn main() {
    // Test simple expression macro
    let a = double!(21);
    print(a);  // Should print 42

    // Test nested macro usage
    let b = add_one!(99);
    print(b);  // Should print 100
}
