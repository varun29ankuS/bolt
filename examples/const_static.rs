// Test const and static items

const ANSWER: i64 = 42;
const DOUBLED: i64 = ANSWER * 2;
const COMPUTED: i64 = 10 + 5 * 2;

static mut COUNTER: i64 = 0;
static INITIAL_VALUE: i64 = 100;

fn main() {
    // Test const values
    print(ANSWER);     // Should print 42
    print(DOUBLED);    // Should print 84
    print(COMPUTED);   // Should print 20

    // Test static read
    print(INITIAL_VALUE);  // Should print 100

    // Test static mut write and read
    COUNTER = 10;
    print(COUNTER);  // Should print 10
    COUNTER = COUNTER + 5;
    print(COUNTER);  // Should print 15
}
