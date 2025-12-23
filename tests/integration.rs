//! Integration tests for Bolt compiler
//!
//! Runs bolt on example files and verifies output.

use std::process::Command;
use std::path::Path;

/// Run bolt on a file and return (success, stdout, stderr)
fn run_bolt(file: &str) -> (bool, String, String) {
    let output = Command::new("cargo")
        .args(["run", "--", "run", file])
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
