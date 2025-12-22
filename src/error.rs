//! Error types for Bolt compiler

use serde::{Deserialize, Serialize};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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

// ============================================================================
// LLM-Friendly JSON Diagnostic Output (BOL-15)
// ============================================================================

/// Error code categories for LLM consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    // Parse errors (E0001-E0099)
    UnexpectedToken,
    UnterminatedString,
    InvalidLiteral,
    MissingSemicolon,
    UnmatchedBrace,

    // Type errors (E0100-E0199)
    TypeMismatch,
    UndefinedVariable,
    UndefinedFunction,
    UndefinedType,
    ArgumentCountMismatch,
    InvalidOperator,
    CannotInferType,

    // Borrow errors (E0200-E0299)
    UseAfterMove,
    BorrowOfMovedValue,
    MutableBorrowConflict,
    ImmutableBorrowConflict,
    CannotMutateImmutable,
    DanglingReference,

    // Lifetime errors (E0300-E0399)
    LifetimeMismatch,
    MissingLifetime,

    // Pattern errors (E0400-E0499)
    NonExhaustivePatterns,
    UnreachablePattern,

    // Codegen errors (E0500-E0599)
    UnsupportedFeature,
    InternalError,

    // Generic fallback
    Unknown,
}

impl ErrorCode {
    /// Get the numeric error code (for compatibility with rustc)
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnexpectedToken => "E0001",
            Self::UnterminatedString => "E0002",
            Self::InvalidLiteral => "E0003",
            Self::MissingSemicolon => "E0004",
            Self::UnmatchedBrace => "E0005",
            Self::TypeMismatch => "E0100",
            Self::UndefinedVariable => "E0101",
            Self::UndefinedFunction => "E0102",
            Self::UndefinedType => "E0103",
            Self::ArgumentCountMismatch => "E0104",
            Self::InvalidOperator => "E0105",
            Self::CannotInferType => "E0106",
            Self::UseAfterMove => "E0200",
            Self::BorrowOfMovedValue => "E0382", // Match rustc
            Self::MutableBorrowConflict => "E0499", // Match rustc
            Self::ImmutableBorrowConflict => "E0502", // Match rustc
            Self::CannotMutateImmutable => "E0596", // Match rustc
            Self::DanglingReference => "E0106",
            Self::LifetimeMismatch => "E0300",
            Self::MissingLifetime => "E0301",
            Self::NonExhaustivePatterns => "E0004", // Match rustc
            Self::UnreachablePattern => "E0001",
            Self::UnsupportedFeature => "E0500",
            Self::InternalError => "E0501",
            Self::Unknown => "E9999",
        }
    }
}

/// Source location with resolved file path and line/column info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// Full file path
    pub file: PathBuf,
    /// 1-indexed line number
    pub line: usize,
    /// 1-indexed column number
    pub column: usize,
    /// End line (for multi-line spans)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    /// End column
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_column: Option<usize>,
}

/// A labeled source span (for showing multiple related locations)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    /// Human-readable label for this span
    pub message: String,
    /// Location in source
    pub location: SourceLocation,
    /// Whether this is the primary span
    #[serde(default)]
    pub primary: bool,
}

/// Suggested fix that can be automatically applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    /// What the suggestion does
    pub message: String,
    /// The replacement text
    pub replacement: String,
    /// Location to apply the fix
    pub location: SourceLocation,
    /// Confidence level (high = safe to auto-apply)
    #[serde(default)]
    pub confidence: SuggestionConfidence,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SuggestionConfidence {
    /// Safe to auto-apply
    High,
    /// Likely correct but review recommended
    #[default]
    Medium,
    /// Possible fix, needs human review
    Low,
}

/// Rich diagnostic output optimized for LLM consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonDiagnostic {
    /// Severity level
    pub level: DiagnosticLevel,
    /// Error code for categorization
    pub code: ErrorCode,
    /// Human-readable error message
    pub message: String,
    /// Primary source location
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
    /// The actual source code at the error location
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_snippet: Option<String>,
    /// Additional labeled spans (e.g., "value moved here", "borrow occurs here")
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<Label>,
    /// Explanatory notes
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// Suggested fixes
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<Suggestion>,
    /// Help text for understanding the error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub help: Option<String>,
}

impl JsonDiagnostic {
    pub fn error(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Error,
            code,
            message: message.into(),
            location: None,
            source_snippet: None,
            labels: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
            help: None,
        }
    }

    pub fn warning(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            code,
            message: message.into(),
            location: None,
            source_snippet: None,
            labels: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
            help: None,
        }
    }

    pub fn with_location(mut self, loc: SourceLocation) -> Self {
        self.location = Some(loc);
        self
    }

    pub fn with_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.source_snippet = Some(snippet.into());
        self
    }

    pub fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn with_notes_from_vec(mut self, notes: Vec<String>) -> Self {
        self.notes.extend(notes);
        self
    }

    /// Serialize to JSON for LLM consumption
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Serialize to pretty JSON for human debugging
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Complete diagnostic report (for batch output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    /// Compiler version
    pub compiler_version: String,
    /// All diagnostics
    pub diagnostics: Vec<JsonDiagnostic>,
    /// Summary statistics
    pub summary: DiagnosticSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticSummary {
    pub errors: usize,
    pub warnings: usize,
    pub notes: usize,
}

impl DiagnosticReport {
    pub fn new(diagnostics: Vec<JsonDiagnostic>) -> Self {
        let errors = diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Error).count();
        let warnings = diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Warning).count();
        let notes = diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Note).count();

        Self {
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            diagnostics,
            summary: DiagnosticSummary { errors, warnings, notes },
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}
