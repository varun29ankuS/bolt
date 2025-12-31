//! Error types for Bolt compiler

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

// ============================================================================
// Source Map for Line/Column Resolution
// ============================================================================

/// Stores source file contents and provides line/column lookup
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    pub content: String,
    /// Byte offset of each line start
    line_starts: Vec<usize>,
}

impl SourceFile {
    pub fn new(path: PathBuf, content: String) -> Self {
        let mut line_starts = vec![0];
        for (i, c) in content.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self { path, content, line_starts }
    }

    /// Convert byte offset to (line, column), both 1-indexed
    pub fn offset_to_line_col(&self, offset: u32) -> (usize, usize) {
        let offset = offset as usize;
        // Binary search for the line
        let line = match self.line_starts.binary_search(&offset) {
            Ok(line) => line,
            Err(line) => line.saturating_sub(1),
        };
        let line_start = self.line_starts.get(line).copied().unwrap_or(0);
        let col = offset.saturating_sub(line_start);
        (line + 1, col + 1) // 1-indexed
    }

    /// Convert (line, column) to byte offset. Both are 1-indexed.
    pub fn line_col_to_offset(&self, line: usize, col: usize) -> u32 {
        if line == 0 {
            return 0;
        }
        let line_idx = line.saturating_sub(1);
        let line_start = self.line_starts.get(line_idx).copied().unwrap_or(0);
        let col_offset = col.saturating_sub(1);
        (line_start + col_offset) as u32
    }

    /// Get the source line at the given 1-indexed line number
    pub fn get_line(&self, line: usize) -> Option<&str> {
        if line == 0 { return None; }
        let line_idx = line - 1;
        let start = self.line_starts.get(line_idx).copied()?;
        let end = self.line_starts.get(line_idx + 1)
            .copied()
            .unwrap_or(self.content.len());
        // Trim trailing newline
        let line_content = &self.content[start..end];
        Some(line_content.trim_end_matches('\n').trim_end_matches('\r'))
    }

    /// Get a snippet of source code around a span
    pub fn get_snippet(&self, start_offset: u32, end_offset: u32) -> String {
        let (start_line, _) = self.offset_to_line_col(start_offset);
        let (end_line, _) = self.offset_to_line_col(end_offset);

        let mut lines = Vec::new();
        for line_num in start_line..=end_line {
            if let Some(line) = self.get_line(line_num) {
                lines.push(format!("{:4} | {}", line_num, line));
            }
        }
        lines.join("\n")
    }

    /// Get source location from span
    pub fn span_to_location(&self, span: Span) -> SourceLocation {
        let (line, column) = self.offset_to_line_col(span.start);
        let (end_line, end_column) = self.offset_to_line_col(span.end);
        SourceLocation {
            file: self.path.clone(),
            line,
            column,
            end_line: if end_line != line { Some(end_line) } else { None },
            end_column: if end_line != line || end_column != column { Some(end_column) } else { None },
        }
    }
}

/// Global source map that tracks all loaded files
pub struct SourceMap {
    files: RwLock<HashMap<u32, Arc<SourceFile>>>,
    path_to_id: RwLock<HashMap<PathBuf, u32>>,
    next_id: RwLock<u32>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self {
            files: RwLock::new(HashMap::new()),
            path_to_id: RwLock::new(HashMap::new()),
            next_id: RwLock::new(1),
        }
    }

    /// Load a source file and return its file_id
    pub fn load_file(&self, path: &PathBuf) -> std::io::Result<u32> {
        // Check if already loaded
        if let Some(&id) = self.path_to_id.read().get(path) {
            return Ok(id);
        }

        let content = std::fs::read_to_string(path)?;
        let file = Arc::new(SourceFile::new(path.clone(), content));

        let mut next_id = self.next_id.write();
        let id = *next_id;
        *next_id += 1;

        self.files.write().insert(id, file);
        self.path_to_id.write().insert(path.clone(), id);

        Ok(id)
    }

    /// Add source content directly (for testing or generated code)
    pub fn add_source(&self, path: PathBuf, content: String) -> u32 {
        let mut next_id = self.next_id.write();
        let id = *next_id;
        *next_id += 1;

        let file = Arc::new(SourceFile::new(path.clone(), content));
        self.files.write().insert(id, file);
        self.path_to_id.write().insert(path, id);

        id
    }

    /// Get source file by ID
    pub fn get_file(&self, file_id: u32) -> Option<Arc<SourceFile>> {
        self.files.read().get(&file_id).cloned()
    }

    /// Get file ID by path
    pub fn get_file_id(&self, path: &PathBuf) -> Option<u32> {
        self.path_to_id.read().get(path).copied()
    }

    /// Convert a span to a source location
    pub fn span_to_location(&self, span: Span) -> Option<SourceLocation> {
        let file = self.get_file(span.file_id)?;
        Some(file.span_to_location(span))
    }

    /// Get source snippet for a span
    pub fn get_snippet(&self, span: Span) -> Option<String> {
        let file = self.get_file(span.file_id)?;
        Some(file.get_snippet(span.start, span.end))
    }

    /// Get the line of source code containing the span
    pub fn get_line_for_span(&self, span: Span) -> Option<String> {
        let file = self.get_file(span.file_id)?;
        let (line, _) = file.offset_to_line_col(span.start);
        file.get_line(line).map(|s| s.to_string())
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Global source map instance
static SOURCE_MAP: once_cell::sync::Lazy<SourceMap> =
    once_cell::sync::Lazy::new(SourceMap::new);

/// Get the global source map
pub fn source_map() -> &'static SourceMap {
    &SOURCE_MAP
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

    // IO errors (E0600-E0699)
    IoError,
    FileNotFound,
    ProjectNotFound,

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
            Self::IoError => "E0600",
            Self::FileNotFound => "E0601",
            Self::ProjectNotFound => "E0602",
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

    /// Convert a simple Diagnostic to a rich JsonDiagnostic using the source map
    pub fn from_diagnostic(diag: Diagnostic, code: ErrorCode) -> Self {
        let mut json_diag = if diag.level == DiagnosticLevel::Error {
            Self::error(code, &diag.message)
        } else {
            Self::warning(code, &diag.message)
        };

        // Add source location and snippet if span is available
        if let Some(span) = diag.span {
            if let Some(loc) = source_map().span_to_location(span) {
                // Get the source line for snippet
                if let Some(line_content) = source_map().get_line_for_span(span) {
                    json_diag.source_snippet = Some(format!(
                        "{:4} | {}\n     | {}{}",
                        loc.line,
                        line_content,
                        " ".repeat(loc.column.saturating_sub(1)),
                        "^".repeat((span.end - span.start).max(1) as usize)
                    ));
                }
                json_diag.location = Some(loc.clone());

                // Add suggestions based on error code
                match code {
                    ErrorCode::BorrowOfMovedValue | ErrorCode::UseAfterMove => {
                        // Extract variable name from message (format: "use of moved value: `name`")
                        let var_name = diag.message
                            .split('`')
                            .nth(1)
                            .unwrap_or("value");
                        json_diag.help = Some(format!(
                            "consider cloning `{}` if you need to use it after the move",
                            var_name
                        ));
                        json_diag.suggestions.push(Suggestion {
                            message: "clone the value before moving".to_string(),
                            replacement: format!("{}.clone()", var_name),
                            location: loc,
                            confidence: SuggestionConfidence::Medium,
                        });
                    }
                    ErrorCode::MutableBorrowConflict => {
                        json_diag.help = Some(
                            "consider using a block to limit the scope of the first borrow".to_string()
                        );
                    }
                    ErrorCode::ImmutableBorrowConflict => {
                        json_diag.help = Some(
                            "cannot borrow as mutable while borrowed as immutable".to_string()
                        );
                    }
                    ErrorCode::CannotMutateImmutable => {
                        json_diag.help = Some(
                            "consider changing this to `let mut` to make it mutable".to_string()
                        );
                    }
                    _ => {}
                }
            }
        }

        // Add notes
        json_diag.notes = diag.notes;

        json_diag
    }

    /// Create a rich borrow error with move information
    pub fn borrow_error(
        message: &str,
        error_span: Option<Span>,
        moved_span: Option<Span>,
        type_name: Option<&str>,
    ) -> Self {
        let mut diag = Self::error(ErrorCode::BorrowOfMovedValue, message);

        // Add primary location
        if let Some(span) = error_span {
            if let Some(loc) = source_map().span_to_location(span) {
                if let Some(line) = source_map().get_line_for_span(span) {
                    diag.source_snippet = Some(format!(
                        "{:4} | {}\n     | {}{}",
                        loc.line,
                        line,
                        " ".repeat(loc.column.saturating_sub(1)),
                        "^".repeat((span.end - span.start).max(1) as usize)
                    ));
                }
                diag.location = Some(loc);
            }
        }

        // Add label for where the move occurred
        if let Some(span) = moved_span {
            if let Some(loc) = source_map().span_to_location(span) {
                diag.labels.push(Label {
                    message: "value moved here".to_string(),
                    location: loc,
                    primary: false,
                });
            }
        }

        // Add type information
        if let Some(ty) = type_name {
            diag.notes.push(format!(
                "move occurs because `{}` has type `{}`, which does not implement `Copy`",
                message.split('`').nth(1).unwrap_or("value"),
                ty
            ));
        }

        // Add suggestion
        diag.help = Some("consider cloning the value if the performance cost is acceptable".to_string());
        if let Some(span) = moved_span {
            if let Some(loc) = source_map().span_to_location(span) {
                diag.suggestions.push(Suggestion {
                    message: "consider cloning the value".to_string(),
                    replacement: ".clone()".to_string(),
                    location: loc,
                    confidence: SuggestionConfidence::Medium,
                });
            }
        }

        diag
    }

    /// Format for human-readable terminal output (colored)
    pub fn format_human(&self) -> String {
        let mut output = String::new();

        // Error header with code
        let level_str = match self.level {
            DiagnosticLevel::Error => "error",
            DiagnosticLevel::Warning => "warning",
            DiagnosticLevel::Note => "note",
        };
        output.push_str(&format!("{}[{}]: {}\n", level_str, self.code.code(), self.message));

        // Location
        if let Some(ref loc) = self.location {
            output.push_str(&format!(" --> {}:{}:{}\n", loc.file.display(), loc.line, loc.column));
        }

        // Source snippet
        if let Some(ref snippet) = self.source_snippet {
            output.push_str("  |\n");
            for line in snippet.lines() {
                output.push_str(&format!("  {}\n", line));
            }
            output.push_str("  |\n");
        }

        // Labels
        for label in &self.labels {
            output.push_str(&format!(
                "  = {}: {}:{}:{}\n",
                label.message,
                label.location.file.display(),
                label.location.line,
                label.location.column
            ));
        }

        // Notes
        for note in &self.notes {
            output.push_str(&format!("  = note: {}\n", note));
        }

        // Help
        if let Some(ref help) = self.help {
            output.push_str(&format!("  = help: {}\n", help));
        }

        // Suggestions
        for suggestion in &self.suggestions {
            output.push_str(&format!(
                "  = suggestion: {} `{}`\n",
                suggestion.message,
                suggestion.replacement
            ));
        }

        output
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

// ============================================================================
// Compilation Progress Reporting (LLM-Friendly)
// ============================================================================

/// Compilation phase for progress tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompilationPhase {
    Parsing,
    TypeCollection,
    TypeChecking,
    BorrowChecking,
    MonomorphizationCollection,
    MonomorphizationCodegen,
    Codegen,
    Linking,
    Execution,
}

impl CompilationPhase {
    pub fn description(&self) -> &'static str {
        match self {
            Self::Parsing => "Parsing source code",
            Self::TypeCollection => "Collecting type definitions",
            Self::TypeChecking => "Type checking",
            Self::BorrowChecking => "Borrow checking",
            Self::MonomorphizationCollection => "Collecting generic instantiations",
            Self::MonomorphizationCodegen => "Generating monomorphized functions",
            Self::Codegen => "Generating machine code",
            Self::Linking => "Linking",
            Self::Execution => "Executing",
        }
    }

    pub fn is_parallelizable(&self) -> bool {
        matches!(
            self,
            Self::TypeCollection | Self::MonomorphizationCodegen | Self::Codegen
        )
    }
}

/// Progress update for LLM consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    /// Current compilation phase
    pub phase: CompilationPhase,
    /// Human-readable status message
    pub message: String,
    /// Progress percentage (0-100, None if indeterminate)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percent: Option<u8>,
    /// Current item being processed (e.g., function name)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_item: Option<String>,
    /// Total items in this phase
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_items: Option<usize>,
    /// Items completed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_items: Option<usize>,
    /// Whether this phase runs in parallel
    #[serde(default)]
    pub parallel: bool,
    /// Elapsed time in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
}

impl ProgressUpdate {
    pub fn new(phase: CompilationPhase) -> Self {
        Self {
            phase,
            message: phase.description().to_string(),
            percent: None,
            current_item: None,
            total_items: None,
            completed_items: None,
            parallel: phase.is_parallelizable(),
            elapsed_ms: None,
        }
    }

    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = msg.into();
        self
    }

    pub fn with_progress(mut self, completed: usize, total: usize) -> Self {
        self.completed_items = Some(completed);
        self.total_items = Some(total);
        self.percent = if total > 0 {
            Some(((completed as f64 / total as f64) * 100.0) as u8)
        } else {
            None
        };
        self
    }

    pub fn with_item(mut self, item: impl Into<String>) -> Self {
        self.current_item = Some(item.into());
        self
    }

    pub fn with_elapsed(mut self, ms: u64) -> Self {
        self.elapsed_ms = Some(ms);
        self
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Complete compilation result for LLM consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationResult {
    /// Whether compilation succeeded
    pub success: bool,
    /// File that was compiled
    pub file: PathBuf,
    /// Phases that were executed
    pub phases_completed: Vec<CompilationPhase>,
    /// Total compilation time in milliseconds
    pub total_time_ms: u64,
    /// Execution result (if code was run)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_result: Option<i64>,
    /// All diagnostics
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<JsonDiagnostic>,
    /// Summary
    pub summary: CompilationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationSummary {
    pub errors: usize,
    pub warnings: usize,
    pub functions_compiled: usize,
    pub types_registered: usize,
}

impl CompilationResult {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}
