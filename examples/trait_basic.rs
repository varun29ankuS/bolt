// Basic trait test

trait Speak {
    fn speak(&self) -> i32;
}

struct Dog {
    volume: i32,
}

struct Cat {
    volume: i32,
}

impl Speak for Dog {
    fn speak(&self) -> i32 {
        self.volume * 2  // Dogs are loud
    }
}

impl Speak for Cat {
    fn speak(&self) -> i32 {
        self.volume  // Cats are quiet
    }
}

fn main() {
    let dog = Dog { volume: 10 };
    let cat = Cat { volume: 5 };

    print(dog.speak());  // 20
    print(cat.speak());  // 5
}
