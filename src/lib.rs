//! Bolt - Lightning-fast Rust compiler for development
//!
//! Architecture:
//! ```text
//! Source -> Parse -> Type Check -> Borrow Check -> Codegen -> Execute/Binary
//!             |          |             |              |
//!          parallel   parallel      parallel       parallel
//! ```

pub mod lexer;
pub mod parser;
pub mod parser2;
pub mod ty;
pub mod typeck;
pub mod borrowck;
pub mod codegen;
pub mod cache;
pub mod cli;
pub mod error;
pub mod hir;
pub mod runtime;

pub use error::{BoltError, Result};
pub use ty::TypeRegistry;
