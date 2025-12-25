// Test dyn Trait (trait objects)

trait Speaker {
    fn speak(&self) -> i64;
}

struct Dog {
    bark_count: i64,
}

impl Speaker for Dog {
    fn speak(&self) -> i64 {
        self.bark_count
    }
}

struct Cat {
    meow_count: i64,
}

impl Speaker for Cat {
    fn speak(&self) -> i64 {
        self.meow_count * 2
    }
}

// Function taking a trait object reference
fn make_noise(s: &dyn Speaker) -> i64 {
    s.speak()
}

fn main() {
    let dog = Dog { bark_count: 21 };
    let cat = Cat { meow_count: 25 };

    // Call through trait object
    print(make_noise(&dog));  // Should print 21
    print(make_noise(&cat));  // Should print 50
}
