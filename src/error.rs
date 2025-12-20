//! Error types for Bolt compiler

use std::path::PathBuf;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, BoltError>;

#[derive(Error, Debug)]
pub enum BoltError {
    #[error("Parse error at {file}:{line}:{col}: {message}")]
    Parse {
        file: PathBuf,
        line: usize,
        col: usize,
        message: String,
    },

    #[error("Type error: {message}")]
    Type { message: String, span: Option<Span> },

    #[error("Borrow checker error: {message}")]
    Borrow { message: String, span: Option<Span> },

    #[error("Code generation error: {message}")]
    Codegen { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    #[error("Internal compiler error: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub file_id: u32,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(file_id: u32, start: u32, end: u32) -> Self {
        Self { file_id, start, end }
    }

    pub fn dummy() -> Self {
        Self {
            file_id: 0,
            start: 0,
            end: 0,
        }
    }

    pub fn merge(self, other: Self) -> Self {
        debug_assert_eq!(self.file_id, other.file_id);
        Self {
            file_id: self.file_id,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Debug)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Note,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Error,
            message: message.into(),
            span: None,
            notes: Vec::new(),
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            message: message.into(),
            span: None,
            notes: Vec::new(),
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

pub struct DiagnosticEmitter {
    diagnostics: Vec<Diagnostic>,
    error_count: usize,
}

impl DiagnosticEmitter {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            error_count: 0,
        }
    }

    pub fn emit(&mut self, diag: Diagnostic) {
        if diag.level == DiagnosticLevel::Error {
            self.error_count += 1;
        }
        self.diagnostics.push(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    pub fn error_count(&self) -> usize {
        self.error_count
    }

    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.error_count = 0;
        std::mem::take(&mut self.diagnostics)
    }
}

impl Default for DiagnosticEmitter {
    fn default() -> Self {
        Self::new()
    }
}
