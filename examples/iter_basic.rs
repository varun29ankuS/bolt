// Basic iterator test - for loop over range

fn main() {
    let mut sum = 0;
    for i in 0..5 {
        sum = sum + i;
    }
    print(sum);  // 0+1+2+3+4 = 10
}
