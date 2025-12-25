// Test control flow constructs

fn main() {
    // Simple if/else
    let x = 10;
    if x > 5 {
        print(1);
    } else {
        print(0);
    }

    // if as expression
    let y = if x > 5 { 100 } else { 200 };
    print(y);

    // while loop
    let mut i = 0;
    while i < 3 {
        print(i);
        i = i + 1;
    }

    // for loop
    for k in 0..3 {
        print(k + 10);
    }

    // loop with break
    let mut j = 0;
    loop {
        if j >= 2 {
            break;
        }
        print(j + 20);
        j = j + 1;
    }
    print(99);
}
