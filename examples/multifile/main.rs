// Multi-file compilation test
// main.rs loads helper.rs as a module

mod helper;

fn main() {
    // Call function from external module
    let x = helper::add(10, 20);
    print(x);  // should be 30

    let y = helper::multiply(5, 6);
    print(y);  // should be 30

    let z = helper::get_constant();
    print(z);  // should be 42
}
