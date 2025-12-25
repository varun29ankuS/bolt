//! Bolt Runtime Library
//!
//! Provides runtime functions for built-in types like Vec, String, Box.
//! These functions are called by generated code via FFI.
//!
//! Memory layout:
//! - Vec<T>:  ptr (8) + len (8) + cap (8) = 24 bytes
//! - String:  ptr (8) + len (8) + cap (8) = 24 bytes
//! - Box<T>:  ptr (8) = 8 bytes
//! - Option<T>: discriminant (8) + T
//! - Result<T,E>: discriminant (8) + max(T, E)

use std::alloc::{alloc, alloc_zeroed, dealloc, realloc, Layout};
use std::ptr;

// ============================================================================
// Memory Allocation (already exists, but here for reference)
// ============================================================================

/// Allocate memory with the system allocator
#[no_mangle]
pub extern "C" fn bolt_alloc(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return align as *mut u8; // Return aligned dangling pointer
    }
    let layout = Layout::from_size_align(size, align).expect("Invalid layout");
    unsafe { alloc(layout) }
}

/// Allocate zeroed memory
#[no_mangle]
pub extern "C" fn bolt_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return align as *mut u8;
    }
    let layout = Layout::from_size_align(size, align).expect("Invalid layout");
    unsafe { alloc_zeroed(layout) }
}

/// Deallocate memory
#[no_mangle]
pub extern "C" fn bolt_dealloc(ptr: *mut u8, size: usize, align: usize) {
    if size == 0 || ptr.is_null() {
        return;
    }
    let layout = Layout::from_size_align(size, align).expect("Invalid layout");
    unsafe { dealloc(ptr, layout) }
}

/// Reallocate memory
#[no_mangle]
pub extern "C" fn bolt_realloc(ptr: *mut u8, old_size: usize, new_size: usize, align: usize) -> *mut u8 {
    if old_size == 0 {
        return bolt_alloc(new_size, align);
    }
    if new_size == 0 {
        bolt_dealloc(ptr, old_size, align);
        return align as *mut u8;
    }
    let layout = Layout::from_size_align(old_size, align).expect("Invalid layout");
    unsafe { realloc(ptr, layout, new_size) }
}

// ============================================================================
// Vec<T> Operations
// ============================================================================

/// Vec representation in memory
#[repr(C)]
pub struct BoltVec {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
}

/// Create a new empty Vec with given element size
#[no_mangle]
pub extern "C" fn bolt_vec_new() -> BoltVec {
    BoltVec {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    }
}

/// Create a Vec with pre-allocated capacity
#[no_mangle]
pub extern "C" fn bolt_vec_with_capacity(capacity: usize, elem_size: usize) -> BoltVec {
    if capacity == 0 || elem_size == 0 {
        return bolt_vec_new();
    }
    let ptr = bolt_alloc(capacity * elem_size, 8);
    BoltVec {
        ptr,
        len: 0,
        cap: capacity,
    }
}

/// Push an element to a Vec (returns new Vec state)
/// elem_ptr points to the element to push
#[no_mangle]
pub extern "C" fn bolt_vec_push(vec: *mut BoltVec, elem_ptr: *const u8, elem_size: usize) {
    unsafe {
        let v = &mut *vec;

        // Grow if needed
        if v.len >= v.cap {
            let new_cap = if v.cap == 0 { 4 } else { v.cap * 2 };
            let new_ptr = bolt_realloc(v.ptr, v.cap * elem_size, new_cap * elem_size, 8);
            v.ptr = new_ptr;
            v.cap = new_cap;
        }

        // Copy element to the end
        let dest = v.ptr.add(v.len * elem_size);
        ptr::copy_nonoverlapping(elem_ptr, dest, elem_size);
        v.len += 1;
    }
}

/// Get a pointer to an element at index
#[no_mangle]
pub extern "C" fn bolt_vec_get(vec: *const BoltVec, index: usize, elem_size: usize) -> *const u8 {
    unsafe {
        let v = &*vec;
        if index >= v.len {
            panic!("Vec index out of bounds: {} >= {}", index, v.len);
        }
        v.ptr.add(index * elem_size)
    }
}

/// Get a mutable pointer to an element at index
#[no_mangle]
pub extern "C" fn bolt_vec_get_mut(vec: *mut BoltVec, index: usize, elem_size: usize) -> *mut u8 {
    unsafe {
        let v = &*vec;
        if index >= v.len {
            panic!("Vec index out of bounds: {} >= {}", index, v.len);
        }
        v.ptr.add(index * elem_size)
    }
}

/// Get the length of a Vec
#[no_mangle]
pub extern "C" fn bolt_vec_len(vec: *const BoltVec) -> usize {
    unsafe { (*vec).len }
}

/// Get the capacity of a Vec
#[no_mangle]
pub extern "C" fn bolt_vec_capacity(vec: *const BoltVec) -> usize {
    unsafe { (*vec).cap }
}

/// Check if a Vec is empty
#[no_mangle]
pub extern "C" fn bolt_vec_is_empty(vec: *const BoltVec) -> bool {
    unsafe { (*vec).len == 0 }
}

/// Pop an element from the end (returns pointer to removed element, or null if empty)
#[no_mangle]
pub extern "C" fn bolt_vec_pop(vec: *mut BoltVec, elem_size: usize) -> *mut u8 {
    unsafe {
        let v = &mut *vec;
        if v.len == 0 {
            return ptr::null_mut();
        }
        v.len -= 1;
        v.ptr.add(v.len * elem_size)
    }
}

/// Clear the Vec (keeps capacity)
#[no_mangle]
pub extern "C" fn bolt_vec_clear(vec: *mut BoltVec) {
    unsafe {
        (*vec).len = 0;
    }
}

/// Drop/free a Vec
#[no_mangle]
pub extern "C" fn bolt_vec_drop(vec: *mut BoltVec, elem_size: usize) {
    unsafe {
        let v = &*vec;
        if !v.ptr.is_null() && v.cap > 0 {
            bolt_dealloc(v.ptr, v.cap * elem_size, 8);
        }
    }
}

/// Create a Vec from a slice (ptr + len)
#[no_mangle]
pub extern "C" fn bolt_vec_from_slice(src_ptr: *const u8, len: usize, elem_size: usize) -> BoltVec {
    if len == 0 {
        return bolt_vec_new();
    }
    let ptr = bolt_alloc(len * elem_size, 8);
    unsafe {
        ptr::copy_nonoverlapping(src_ptr, ptr, len * elem_size);
    }
    BoltVec {
        ptr,
        len,
        cap: len,
    }
}

// ============================================================================
// String Operations
// ============================================================================

/// String representation (same as Vec<u8>)
pub type BoltString = BoltVec;

/// Create a new empty String
#[no_mangle]
pub extern "C" fn bolt_string_new() -> BoltString {
    bolt_vec_new()
}

/// Create a String from a UTF-8 byte slice
#[no_mangle]
pub extern "C" fn bolt_string_from_utf8(ptr: *const u8, len: usize) -> BoltString {
    bolt_vec_from_slice(ptr, len, 1)
}

/// Get the length of a String in bytes
#[no_mangle]
pub extern "C" fn bolt_string_len(s: *const BoltString) -> usize {
    bolt_vec_len(s)
}

/// Get a pointer to the String's bytes
#[no_mangle]
pub extern "C" fn bolt_string_as_ptr(s: *const BoltString) -> *const u8 {
    unsafe { (*s).ptr }
}

/// Push a byte to the String
#[no_mangle]
pub extern "C" fn bolt_string_push_byte(s: *mut BoltString, byte: u8) {
    let b = byte;
    bolt_vec_push(s, &b as *const u8, 1);
}

/// Push a str to the String
#[no_mangle]
pub extern "C" fn bolt_string_push_str(s: *mut BoltString, ptr: *const u8, len: usize) {
    for i in 0..len {
        unsafe {
            let byte = *ptr.add(i);
            bolt_string_push_byte(s, byte);
        }
    }
}

/// Drop/free a String
#[no_mangle]
pub extern "C" fn bolt_string_drop(s: *mut BoltString) {
    bolt_vec_drop(s, 1);
}

// ============================================================================
// Box<T> Operations
// ============================================================================

/// Allocate a Box<T>
#[no_mangle]
pub extern "C" fn bolt_box_new(size: usize, align: usize) -> *mut u8 {
    bolt_alloc(size, align)
}

/// Drop/free a Box<T>
#[no_mangle]
pub extern "C" fn bolt_box_drop(ptr: *mut u8, size: usize, align: usize) {
    bolt_dealloc(ptr, size, align);
}

// ============================================================================
// I/O Operations
// ============================================================================

/// Print an integer to stdout
#[no_mangle]
pub extern "C" fn bolt_print_i64(value: i64) {
    println!("{}", value);
}

/// Print an unsigned integer to stdout
#[no_mangle]
pub extern "C" fn bolt_print_u64(value: u64) {
    println!("{}", value);
}

/// Print a float to stdout
#[no_mangle]
pub extern "C" fn bolt_print_f64(value: f64) {
    println!("{}", value);
}

/// Print a string (ptr + len) to stdout
#[no_mangle]
pub extern "C" fn bolt_print_str(ptr: *const u8, len: usize) {
    if ptr.is_null() || len == 0 {
        println!();
        return;
    }
    unsafe {
        let slice = std::slice::from_raw_parts(ptr, len);
        if let Ok(s) = std::str::from_utf8(slice) {
            print!("{}", s);
        }
    }
}

/// Print a newline
#[no_mangle]
pub extern "C" fn bolt_print_newline() {
    println!();
}

/// Print a bool
#[no_mangle]
pub extern "C" fn bolt_print_bool(value: bool) {
    println!("{}", value);
}

/// Print a char
#[no_mangle]
pub extern "C" fn bolt_print_char(value: u32) {
    if let Some(c) = char::from_u32(value) {
        println!("{}", c);
    }
}

// ============================================================================
// Panic/Abort
// ============================================================================

/// Panic with a message
#[no_mangle]
pub extern "C" fn bolt_panic(msg_ptr: *const u8, msg_len: usize) -> ! {
    let msg = if msg_ptr.is_null() || msg_len == 0 {
        "explicit panic".to_string()
    } else {
        unsafe {
            let slice = std::slice::from_raw_parts(msg_ptr, msg_len);
            std::str::from_utf8(slice).unwrap_or("invalid utf8 in panic message").to_string()
        }
    };
    panic!("{}", msg);
}

/// Abort the program
#[no_mangle]
pub extern "C" fn bolt_abort() -> ! {
    std::process::abort();
}

// ============================================================================
// Iterator Support (basic)
// ============================================================================

/// Iterator state for Vec iteration
#[repr(C)]
pub struct BoltVecIter {
    pub vec_ptr: *const BoltVec,
    pub index: usize,
    pub elem_size: usize,
}

/// Create an iterator over a Vec
#[no_mangle]
pub extern "C" fn bolt_vec_iter(vec: *const BoltVec, elem_size: usize) -> BoltVecIter {
    BoltVecIter {
        vec_ptr: vec,
        index: 0,
        elem_size,
    }
}

/// Get next element from iterator (returns null when exhausted)
#[no_mangle]
pub extern "C" fn bolt_vec_iter_next(iter: *mut BoltVecIter) -> *const u8 {
    unsafe {
        let it = &mut *iter;
        let vec = &*it.vec_ptr;
        if it.index >= vec.len {
            return ptr::null();
        }
        let ptr = vec.ptr.add(it.index * it.elem_size);
        it.index += 1;
        ptr
    }
}

// ============================================================================
// Sum for numeric iterators
// ============================================================================

/// Sum all i64 elements in a Vec
#[no_mangle]
pub extern "C" fn bolt_vec_sum_i64(vec: *const BoltVec) -> i64 {
    unsafe {
        let v = &*vec;
        let mut sum: i64 = 0;
        for i in 0..v.len {
            let ptr = v.ptr.add(i * 8) as *const i64;
            sum += *ptr;
        }
        sum
    }
}

/// Sum all i32 elements in a Vec
#[no_mangle]
pub extern "C" fn bolt_vec_sum_i32(vec: *const BoltVec) -> i32 {
    unsafe {
        let v = &*vec;
        let mut sum: i32 = 0;
        for i in 0..v.len {
            let ptr = v.ptr.add(i * 4) as *const i32;
            sum += *ptr;
        }
        sum
    }
}

/// Sum all f64 elements in a Vec
#[no_mangle]
pub extern "C" fn bolt_vec_sum_f64(vec: *const BoltVec) -> f64 {
    unsafe {
        let v = &*vec;
        let mut sum: f64 = 0.0;
        for i in 0..v.len {
            let ptr = v.ptr.add(i * 8) as *const f64;
            sum += *ptr;
        }
        sum
    }
}

// ============================================================================
// HashMap<K, V> Operations (K and V are i64 for now)
// ============================================================================

/// Entry state constants
const ENTRY_EMPTY: u64 = 0;
const ENTRY_OCCUPIED: u64 = 1;
const ENTRY_TOMBSTONE: u64 = 2;

/// HashMap entry: state(8) + hash(8) + key(8) + value(8) = 32 bytes
const ENTRY_SIZE: usize = 32;

/// HashMap representation in memory
#[repr(C)]
pub struct BoltHashMap {
    pub entries: *mut u8,  // Pointer to entry array
    pub len: usize,        // Number of occupied entries
    pub cap: usize,        // Total capacity (number of slots)
}

/// Simple hash function for i64 keys (FNV-1a inspired)
fn hash_i64(key: i64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let bytes = key.to_le_bytes();
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Create a new empty HashMap
#[no_mangle]
pub extern "C" fn bolt_hashmap_new() -> BoltHashMap {
    BoltHashMap {
        entries: ptr::null_mut(),
        len: 0,
        cap: 0,
    }
}

/// Create a HashMap with pre-allocated capacity
#[no_mangle]
pub extern "C" fn bolt_hashmap_with_capacity(capacity: usize) -> BoltHashMap {
    if capacity == 0 {
        return bolt_hashmap_new();
    }
    // Round up to power of 2
    let cap = capacity.next_power_of_two().max(8);
    let entries = bolt_alloc_zeroed(cap * ENTRY_SIZE, 8);
    BoltHashMap {
        entries,
        len: 0,
        cap,
    }
}

/// Grow the hashmap when load factor exceeds threshold
unsafe fn bolt_hashmap_grow(map: *mut BoltHashMap) {
    let m = &mut *map;
    let old_cap = m.cap;
    let old_entries = m.entries;

    let new_cap = if old_cap == 0 { 8 } else { old_cap * 2 };
    let new_entries = bolt_alloc_zeroed(new_cap * ENTRY_SIZE, 8);

    m.entries = new_entries;
    m.cap = new_cap;
    m.len = 0;

    // Rehash all entries
    if old_cap > 0 && !old_entries.is_null() {
        for i in 0..old_cap {
            let entry = old_entries.add(i * ENTRY_SIZE);
            let state = *(entry as *const u64);
            if state == ENTRY_OCCUPIED {
                let hash = *(entry.add(8) as *const u64);
                let key = *(entry.add(16) as *const i64);
                let value = *(entry.add(24) as *const i64);
                bolt_hashmap_insert_internal(map, hash, key, value);
            }
        }
        bolt_dealloc(old_entries, old_cap * ENTRY_SIZE, 8);
    }
}

/// Internal insert (used during rehash)
unsafe fn bolt_hashmap_insert_internal(map: *mut BoltHashMap, hash: u64, key: i64, value: i64) {
    let m = &mut *map;
    let mask = m.cap - 1;
    let mut idx = (hash as usize) & mask;

    loop {
        let entry = m.entries.add(idx * ENTRY_SIZE);
        let state = *(entry as *const u64);

        if state == ENTRY_EMPTY || state == ENTRY_TOMBSTONE {
            *(entry as *mut u64) = ENTRY_OCCUPIED;
            *(entry.add(8) as *mut u64) = hash;
            *(entry.add(16) as *mut i64) = key;
            *(entry.add(24) as *mut i64) = value;
            m.len += 1;
            return;
        }

        // Check if key already exists
        let entry_key = *(entry.add(16) as *const i64);
        if state == ENTRY_OCCUPIED && entry_key == key {
            // Update value
            *(entry.add(24) as *mut i64) = value;
            return;
        }

        idx = (idx + 1) & mask;
    }
}

/// Insert a key-value pair
#[no_mangle]
pub extern "C" fn bolt_hashmap_insert(map: *mut BoltHashMap, key: i64, value: i64) {
    unsafe {
        let m = &mut *map;

        // Grow if load factor > 0.75
        if m.cap == 0 || (m.len + 1) * 4 > m.cap * 3 {
            bolt_hashmap_grow(map);
        }

        let hash = hash_i64(key);
        bolt_hashmap_insert_internal(map, hash, key, value);
    }
}

/// Get a value by key (returns pointer to value, or null if not found)
#[no_mangle]
pub extern "C" fn bolt_hashmap_get(map: *const BoltHashMap, key: i64) -> *const i64 {
    unsafe {
        let m = &*map;
        if m.cap == 0 || m.entries.is_null() {
            return ptr::null();
        }

        let hash = hash_i64(key);
        let mask = m.cap - 1;
        let mut idx = (hash as usize) & mask;
        let start = idx;

        loop {
            let entry = m.entries.add(idx * ENTRY_SIZE);
            let state = *(entry as *const u64);

            if state == ENTRY_EMPTY {
                return ptr::null();
            }

            if state == ENTRY_OCCUPIED {
                let entry_key = *(entry.add(16) as *const i64);
                if entry_key == key {
                    return entry.add(24) as *const i64;
                }
            }

            idx = (idx + 1) & mask;
            if idx == start {
                return ptr::null();
            }
        }
    }
}

/// Check if map contains a key
#[no_mangle]
pub extern "C" fn bolt_hashmap_contains_key(map: *const BoltHashMap, key: i64) -> i64 {
    if bolt_hashmap_get(map, key).is_null() { 0 } else { 1 }
}

/// Remove a key from the map (returns the removed value, or 0 if not found)
#[no_mangle]
pub extern "C" fn bolt_hashmap_remove(map: *mut BoltHashMap, key: i64) -> i64 {
    unsafe {
        let m = &mut *map;
        if m.cap == 0 || m.entries.is_null() {
            return 0;
        }

        let hash = hash_i64(key);
        let mask = m.cap - 1;
        let mut idx = (hash as usize) & mask;
        let start = idx;

        loop {
            let entry = m.entries.add(idx * ENTRY_SIZE);
            let state = *(entry as *const u64);

            if state == ENTRY_EMPTY {
                return 0;
            }

            if state == ENTRY_OCCUPIED {
                let entry_key = *(entry.add(16) as *const i64);
                if entry_key == key {
                    let value = *(entry.add(24) as *const i64);
                    *(entry as *mut u64) = ENTRY_TOMBSTONE;
                    m.len -= 1;
                    return value;
                }
            }

            idx = (idx + 1) & mask;
            if idx == start {
                return 0;
            }
        }
    }
}

/// Get the number of entries in the map
#[no_mangle]
pub extern "C" fn bolt_hashmap_len(map: *const BoltHashMap) -> usize {
    unsafe { (*map).len }
}

/// Check if the map is empty
#[no_mangle]
pub extern "C" fn bolt_hashmap_is_empty(map: *const BoltHashMap) -> i64 {
    unsafe { if (*map).len == 0 { 1 } else { 0 } }
}

/// Clear the map (keeps capacity)
#[no_mangle]
pub extern "C" fn bolt_hashmap_clear(map: *mut BoltHashMap) {
    unsafe {
        let m = &mut *map;
        if m.cap > 0 && !m.entries.is_null() {
            ptr::write_bytes(m.entries, 0, m.cap * ENTRY_SIZE);
        }
        m.len = 0;
    }
}

/// Drop/free the map
#[no_mangle]
pub extern "C" fn bolt_hashmap_drop(map: *mut BoltHashMap) {
    unsafe {
        let m = &*map;
        if !m.entries.is_null() && m.cap > 0 {
            bolt_dealloc(m.entries, m.cap * ENTRY_SIZE, 8);
        }
    }
}

// ============================================================================
// Slice Operations (&[T])
// Slices are fat pointers: (ptr: *const T, len: usize) = 16 bytes
// ============================================================================

/// Slice representation in memory (fat pointer)
#[repr(C)]
pub struct BoltSlice {
    pub ptr: *const u8,
    pub len: usize,
}

/// Get the length of a slice
#[no_mangle]
pub extern "C" fn bolt_slice_len(slice_ptr: i64, slice_len: i64) -> i64 {
    slice_len
}

/// Get a pointer to an element at index with bounds checking
#[no_mangle]
pub extern "C" fn bolt_slice_get(slice_ptr: i64, slice_len: i64, index: i64, elem_size: i64) -> i64 {
    if index < 0 || index >= slice_len {
        panic!("Slice index out of bounds: {} not in 0..{}", index, slice_len);
    }
    slice_ptr + index * elem_size
}

/// Get element at index without bounds checking (for when bounds already verified)
#[no_mangle]
pub extern "C" fn bolt_slice_get_unchecked(slice_ptr: i64, index: i64, elem_size: i64) -> i64 {
    slice_ptr + index * elem_size
}

/// Check if index is valid for slice
#[no_mangle]
pub extern "C" fn bolt_slice_bounds_check(slice_len: i64, index: i64) -> i64 {
    if index >= 0 && index < slice_len { 1 } else { 0 }
}

/// Create a slice from an array (returns ptr, length stays the same)
#[no_mangle]
pub extern "C" fn bolt_array_as_slice(array_ptr: i64, len: i64) -> BoltSlice {
    BoltSlice {
        ptr: array_ptr as *const u8,
        len: len as usize,
    }
}

/// Create a subslice (returns new ptr, new len)
#[no_mangle]
pub extern "C" fn bolt_slice_subslice(slice_ptr: i64, slice_len: i64, start: i64, end: i64, elem_size: i64) -> BoltSlice {
    if start < 0 || end > slice_len || start > end {
        panic!("Invalid subslice range: {}..{} for slice of length {}", start, end, slice_len);
    }
    BoltSlice {
        ptr: (slice_ptr + start * elem_size) as *const u8,
        len: (end - start) as usize,
    }
}

/// Check if slice is empty
#[no_mangle]
pub extern "C" fn bolt_slice_is_empty(slice_len: i64) -> i64 {
    if slice_len == 0 { 1 } else { 0 }
}

/// Get first element of slice (panics if empty)
#[no_mangle]
pub extern "C" fn bolt_slice_first(slice_ptr: i64, slice_len: i64) -> i64 {
    if slice_len == 0 {
        panic!("Called first() on empty slice");
    }
    slice_ptr
}

/// Get last element of slice (panics if empty)
#[no_mangle]
pub extern "C" fn bolt_slice_last(slice_ptr: i64, slice_len: i64, elem_size: i64) -> i64 {
    if slice_len == 0 {
        panic!("Called last() on empty slice");
    }
    slice_ptr + (slice_len - 1) * elem_size
}

/// Iterator state for slice iteration
#[repr(C)]
pub struct BoltSliceIter {
    pub ptr: i64,
    pub len: i64,
    pub index: i64,
    pub elem_size: i64,
}

/// Create an iterator over a slice
#[no_mangle]
pub extern "C" fn bolt_slice_iter(slice_ptr: i64, slice_len: i64, elem_size: i64) -> BoltSliceIter {
    BoltSliceIter {
        ptr: slice_ptr,
        len: slice_len,
        index: 0,
        elem_size,
    }
}

/// Get next element from slice iterator (returns 0 when exhausted, else ptr to element)
#[no_mangle]
pub extern "C" fn bolt_slice_iter_next(iter: *mut BoltSliceIter) -> i64 {
    unsafe {
        let it = &mut *iter;
        if it.index >= it.len {
            return 0;
        }
        let elem_ptr = it.ptr + it.index * it.elem_size;
        it.index += 1;
        elem_ptr
    }
}
