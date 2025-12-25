//! Integration tests for Bolt compiler
//!
//! Runs bolt on example files and verifies output.

use std::process::Command;
use std::path::Path;

/// Run bolt on a file and return (success, stdout, stderr)
fn run_bolt(file: &str) -> (bool, String, String) {
    // Use --sync-borrow to avoid persistent cache issues in tests
    let output = Command::new("cargo")
        .args(["run", "--", "run", "--sync-borrow", file])
        .output()
        .expect("Failed to execute bolt");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), stdout, stderr)
}

/// Extract the program output (lines after "Running...")
fn extract_output(stdout: &str) -> Vec<String> {
    let mut in_output = false;
    let mut lines = Vec::new();

    for line in stdout.lines() {
        if line.contains("Running...") {
            in_output = true;
            continue;
        }
        if line.contains("Finished with code") {
            break;
        }
        if in_output && !line.is_empty() {
            lines.push(line.trim().to_string());
        }
    }
    lines
}

/// Test macro for expected output
macro_rules! test_example {
    ($name:ident, $file:expr, $expected:expr) => {
        #[test]
        fn $name() {
            let (success, stdout, stderr) = run_bolt($file);
            if !success {
                eprintln!("STDERR:\n{}", stderr);
                eprintln!("STDOUT:\n{}", stdout);
            }
            assert!(success || stdout.contains("Warning:"), "Compilation failed for {}", $file);

            let output = extract_output(&stdout);
            let expected: Vec<String> = $expected.iter().map(|s: &&str| s.to_string()).collect();
            assert_eq!(output, expected, "Output mismatch for {}", $file);
        }
    };
}

// Basic struct tests
test_example!(test_very_simple, "examples/very_simple.rs", &["42", "99"]);
test_example!(test_ptr_test, "examples/ptr_test.rs", &["10", "200"]);
test_example!(test_static_ctor_simple, "examples/static_ctor_simple.rs", &["42", "10", "200"]);
test_example!(test_method_chain, "examples/method_chain.rs", &["123", "123"]);

// Test that compilation at least succeeds (output may vary)
macro_rules! test_compiles {
    ($name:ident, $file:expr) => {
        #[test]
        fn $name() {
            let (success, stdout, stderr) = run_bolt($file);
            if !success && !stdout.contains("Warning:") {
                eprintln!("STDERR:\n{}", stderr);
                eprintln!("STDOUT:\n{}", stdout);
                panic!("Failed to compile {}", $file);
            }
        }
    };
}

test_compiles!(test_nested_method_compiles, "examples/nested_method.rs");
test_compiles!(test_simple_nested_compiles, "examples/simple_nested.rs");

// Generic tests
test_example!(test_generic_basic, "examples/generic_basic.rs", &["42"]);
test_example!(test_generic_function, "examples/generic_function.rs", &["42", "100", "10"]);
test_example!(test_generic_method, "examples/generic_method.rs", &["42"]);
test_example!(test_generic_pair, "examples/generic_pair.rs", &["10", "20", "100", "400"]);
test_example!(test_generic_nested, "examples/generic_nested.rs", &["42", "100", "200"]);

// Control flow tests
test_example!(test_control_flow, "examples/control_flow.rs", &["1", "100", "0", "1", "2", "10", "11", "12", "20", "21", "99"]);
test_example!(test_match_enum, "examples/match_enum.rs", &["1", "2", "3", "1"]);

// &self methods
test_example!(test_ref_self, "examples/ref_self.rs", &["10"]);

// &mut self methods
test_example!(test_mut_self, "examples/mut_self.rs", &["10", "11", "16"]);

// Option pattern matching
test_example!(test_option_match, "examples/option_match.rs", &["42"]);
test_example!(test_option_if_let, "examples/option_if_let.rs", &["42", "99"]);
test_example!(test_option_shorthand, "examples/option_shorthand.rs", &["42", "42"]);
test_example!(test_option_unwrap, "examples/option_unwrap.rs", &["42", "100", "1", "2"]);

// Result and ? operator
test_example!(test_try_operator, "examples/try_operator.rs", &["20", "-1"]);

// Level 2 nested patterns - Some(Struct{field})
test_example!(test_option_nested, "examples/option_nested.rs", &["30"]);

// Vec<T> basic operations
test_example!(test_vec_basic, "examples/vec_basic.rs", &["3", "10", "20", "30", "30", "2", "100"]);

// String basic operations
test_example!(test_string_basic, "examples/string_basic.rs", &["5", "0", "5", "Hello, Bolt!"]);

// Trait impl dispatch
test_example!(test_trait_basic, "examples/trait_basic.rs", &["20", "5"]);

// Result enum with pattern matching
test_example!(test_result_basic, "examples/result_basic.rs", &["5", "-1"]);

// Tuple creation and field access
test_example!(test_tuple_basic, "examples/tuple_basic.rs", &["10", "20", "6", "30"]);

// Array creation and indexing
test_example!(test_array_basic, "examples/array_basic.rs", &["1", "3", "5"]);

// Trait bounds on generic functions
test_example!(test_trait_bound, "examples/trait_bound.rs", &["42", "15"]);

// Closures with parameters and capture
test_example!(test_closure_basic, "examples/closure_basic.rs", &["10", "30", "105"]);

// Basic module with functions
test_example!(test_module_basic, "examples/module_basic.rs", &["30", "30"]);

// For loop over range
test_example!(test_iter_basic, "examples/iter_basic.rs", &["10"]);

// Nested modules with relative path resolution
test_example!(test_module_nested, "examples/module_nested.rs", &["42", "42"]);

// Structs in modules with methods
test_example!(test_module_struct, "examples/module_struct.rs", &["30"]);

// Vec iterator with for loop
test_example!(test_vec_iter, "examples/vec_iter.rs", &["60"]);

// Array iterator with for loop
test_example!(test_array_iter, "examples/array_iter.rs", &["15"]);

// HashMap basic operations
test_example!(test_hashmap_basic, "examples/hashmap_basic.rs", &["3", "100", "200", "300", "1", "0", "2", "0"]);

// Float (f64) basic operations - tests Copy semantics (no move errors)
test_example!(test_float_basic, "examples/float_basic.rs", &["3.14", "2", "5.140000000000001", "1.1400000000000001", "6.28", "1.57"]);

// Multi-file compilation (mod foo; loads foo.rs)
test_example!(test_multifile, "examples/multifile/main.rs", &["30", "30", "42"]);

// Use statements (use mod::item)
test_example!(test_use_statement, "examples/use_statement.rs", &["15", "12"]);

// Derive Clone macro
test_example!(test_derive_clone, "examples/derive_clone.rs", &["10", "20", "10", "20"]);

// Explicit lifetime annotations
test_example!(test_lifetime_basic, "examples/lifetime_basic.rs", &["42", "100"]);

// Slice types (&[T])
test_example!(test_slice_basic, "examples/slice_basic.rs", &["60", "10", "3"]);

// Associated types in traits
test_example!(test_assoc_type_basic, "examples/assoc_type_basic.rs", &["42", "30"]);

// macro_rules! basic macros
test_example!(test_macro_simple, "examples/macro_simple.rs", &["42"]);
test_example!(test_macro_basic, "examples/macro_basic.rs", &["42", "100"]);

// const and static items
test_example!(test_const_static, "examples/const_static.rs", &["42", "84", "20", "100", "10", "15"]);

// impl Trait return types
test_example!(test_impl_trait, "examples/impl_trait.rs", &["42", "100"]);

// dyn Trait (trait objects) - static dispatch for single-type case
test_example!(test_dyn_trait_simple, "examples/dyn_trait_simple.rs", &["42"]);
