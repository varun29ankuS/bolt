//! Bolt - Lightning-fast Rust type checker for LLM/AI-assisted development
//!
//! Bolt provides instant feedback on Rust code, optimized for automated
//! code generation workflows. Check code in milliseconds instead of seconds.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use bolt_rs::api::{check, CheckConfig};
//!
//! // Check Rust source code
//! let result = check("fn main() { let x: i32 = 42; }", CheckConfig::default());
//! assert!(result.success);
//!
//! // Check with errors
//! let result = check(r#"fn main() { let x: i32 = "hello"; }"#, CheckConfig::default());
//! assert!(!result.success);
//! for error in &result.errors {
//!     println!("Error: {}", error.message);
//!     for fix in &error.fixes {
//!         println!("  Suggested fix: {}", fix.description);
//!     }
//! }
//! ```
//!
//! # Architecture
//!
//! ```text
//! Source -> Parse -> Type Check -> Borrow Check -> Codegen -> Execute
//!             |          |             |              |
//!          parallel   parallel      async          parallel
//! ```
//!
//! # Modules
//!
//! - [`api`] - **Primary interface** for programmatic use
//! - [`cli`] - Command-line interface
//! - [`parser`] - Syn-based Rust parser
//! - [`parser2`] - Chumsky-based parser (experimental)
//! - [`typeck`] - Type checking and inference
//! - [`borrowck`] - Borrow checking (sync and async)
//! - [`codegen`] - Cranelift-based code generation
//! - [`error`] - Error types and diagnostics

// ============================================================================
// Public API (primary interface for library users)
// ============================================================================

pub mod api;

// Re-export main API types at crate root for convenience
pub use api::{
    check, check_file, check_json, check_json_pretty, quick_check, strict_check,
    CheckConfig, CheckResult, Fix, FixKind, Patch, ParseResult, RichDiagnostic,
    Severity, SourceLocation, Strictness, ParserBackend,
};

// ============================================================================
// Internal Modules (exposed for advanced use cases)
// ============================================================================

pub mod borrowck;
pub mod cache;
pub mod cargo;
pub mod cli;
pub mod codegen;
pub mod derive;
pub mod error;
pub mod extern_crates;
pub mod fixes;
pub mod hir;
pub mod incremental;
pub mod lexer;
pub mod lsp;
pub mod parser;
pub mod parser2;
pub mod runtime;
pub mod ty;
pub mod typeck;

// Re-export commonly used internal types
pub use error::{BoltError, Result};
pub use ty::TypeRegistry;
