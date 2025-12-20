//! Bolt - Lightning-fast Rust compiler for development
//!
//! Architecture:
//! ```
//! Source → Parse → Type Check → Borrow Check → Codegen → Execute/Binary
//!            ↓         ↓            ↓             ↓
//!         parallel  parallel     parallel      parallel
//! ```

pub mod parser;
pub mod typeck;
pub mod borrowck;
pub mod codegen;
pub mod cache;
pub mod cli;
pub mod error;
pub mod hir;

pub use error::{BoltError, Result};
