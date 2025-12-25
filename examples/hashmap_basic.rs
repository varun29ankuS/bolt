// HashMap basic operations test

fn main() {
    let map: HashMap<i64, i64> = HashMap::new();

    // Insert some key-value pairs
    map.insert(1, 100);
    map.insert(2, 200);
    map.insert(3, 300);

    // Test len()
    print(map.len());  // should be 3

    // Test get
    print(map.get(1));  // should be 100
    print(map.get(2));  // should be 200
    print(map.get(3));  // should be 300

    // Test contains_key
    let has2 = map.contains_key(2);
    print(has2);       // should be 1 (true)
    let has99 = map.contains_key(99);
    print(has99);      // should be 0 (false)

    // Test remove
    map.remove(2);
    print(map.len());  // should be 2
    let has2_after = map.contains_key(2);
    print(has2_after); // should be 0 (false)
}
