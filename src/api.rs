//! Bolt Public API
//!
//! This module provides the programmatic interface to Bolt, designed for
//! LLM/AI tool integration. All functions return structured, machine-parseable
//! results optimized for automated code generation workflows.
//!
//! # Example
//!
//! ```rust,ignore
//! use bolt_rs::api::{check, CheckConfig, Strictness};
//!
//! let source = r#"
//!     fn main() {
//!         let x: i32 = "hello"; // type error
//!     }
//! "#;
//!
//! let result = check(source, CheckConfig::default());
//! if !result.success {
//!     for error in &result.errors {
//!         println!("Error: {}", error.message);
//!         for fix in &error.fixes {
//!             println!("  Fix: {}", fix.description);
//!         }
//!     }
//! }
//! ```

use crate::borrowck::{BorrowChecker, NllChecker};
use crate::error::{source_map, Diagnostic, ErrorCode, Span};
use crate::hir::Crate;
use crate::ty::TypeRegistry;
use crate::typeck::{TypeChecker, TypeContext};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Configuration Types
// ============================================================================

/// How strict the type checker should be
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Strictness {
    /// Accept more programs, show all errors without cascading failures.
    /// Best for LLM iteration - get maximum feedback per check.
    Lenient,

    /// Standard Rust semantics - match rustc behavior.
    #[default]
    Standard,

    /// Stricter than rustc - catch more potential bugs.
    /// Warns on things rustc allows but are often mistakes.
    Strict,
}

/// Configuration for check operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckConfig {
    /// How strict to be with type checking
    pub strictness: Strictness,

    /// Generate fix suggestions for errors
    pub include_suggestions: bool,

    /// Return partial type information even when errors occur
    pub include_partial_types: bool,

    /// Maximum number of errors to report (0 = unlimited)
    pub max_errors: usize,

    /// Include detailed explanations for errors
    pub include_explanations: bool,

    /// Run borrow checking (can be disabled for faster type-only checks)
    pub run_borrow_check: bool,

    /// Parser backend to use
    pub parser: ParserBackend,
}

impl Default for CheckConfig {
    fn default() -> Self {
        Self {
            strictness: Strictness::Standard,
            include_suggestions: true,
            include_partial_types: true,
            max_errors: 100,
            include_explanations: true,
            run_borrow_check: true,
            parser: ParserBackend::Syn,
        }
    }
}

impl CheckConfig {
    /// Lenient config optimized for LLM iteration
    pub fn lenient() -> Self {
        Self {
            strictness: Strictness::Lenient,
            include_suggestions: true,
            include_partial_types: true,
            max_errors: 0, // Show all errors
            include_explanations: true,
            run_borrow_check: true,
            parser: ParserBackend::Syn,
        }
    }

    /// Strict config that matches rustc behavior
    pub fn strict() -> Self {
        Self {
            strictness: Strictness::Strict,
            include_suggestions: true,
            include_partial_types: false,
            max_errors: 100,
            include_explanations: true,
            run_borrow_check: true,
            parser: ParserBackend::Syn,
        }
    }

    /// Fast config for quick feedback (no borrow check)
    pub fn fast() -> Self {
        Self {
            strictness: Strictness::Lenient,
            include_suggestions: false,
            include_partial_types: false,
            max_errors: 10,
            include_explanations: false,
            run_borrow_check: false,
            parser: ParserBackend::Syn,
        }
    }
}

/// Parser backend selection
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParserBackend {
    /// Syn-based parser (more complete)
    #[default]
    Syn,
    /// Chumsky-based parser (better error recovery)
    Chumsky,
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of a check operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Whether the check passed with no errors
    pub success: bool,

    /// All errors found
    pub errors: Vec<RichDiagnostic>,

    /// All warnings found
    pub warnings: Vec<RichDiagnostic>,

    /// Partial type information (variable -> type string)
    /// Populated even when errors occur if `include_partial_types` is set
    pub partial_types: HashMap<String, String>,

    /// Variables/expressions where type inference gave up
    pub unknowns: Vec<UnknownInfo>,

    /// Overall confidence in the result (0.0 - 1.0)
    /// 1.0 = fully checked, high confidence
    /// 0.5 = some unknowns, may have missed errors
    /// 0.0 = too many unknowns, recommend using rustc
    pub confidence: f32,

    /// Recommendation for the caller
    pub recommendation: Recommendation,

    /// Timing information
    pub timing: TimingInfo,

    /// Statistics about the check
    pub stats: CheckStats,
}

impl CheckResult {
    /// Create a successful result
    pub fn success() -> Self {
        Self {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            partial_types: HashMap::new(),
            unknowns: Vec::new(),
            confidence: 1.0,
            recommendation: Recommendation::TrustBolt,
            timing: TimingInfo::default(),
            stats: CheckStats::default(),
        }
    }

    /// Create a failed result
    pub fn failed(errors: Vec<RichDiagnostic>) -> Self {
        Self {
            success: false,
            errors,
            warnings: Vec::new(),
            partial_types: HashMap::new(),
            unknowns: Vec::new(),
            confidence: 1.0,
            recommendation: Recommendation::TrustBolt,
            timing: TimingInfo::default(),
            stats: CheckStats::default(),
        }
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Convert to pretty JSON string
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// A rich diagnostic with fix suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichDiagnostic {
    /// Error code (E0308, E0382, etc.)
    pub code: String,

    /// Human-readable message
    pub message: String,

    /// Severity level
    pub severity: Severity,

    /// Source location
    pub location: Option<SourceLocation>,

    /// Source code snippet with caret
    pub snippet: Option<String>,

    /// Concrete fix suggestions
    pub fixes: Vec<Fix>,

    /// Detailed explanation of why this is an error
    pub explanation: Option<String>,

    /// Link to Rust documentation
    pub learn_more: Option<String>,

    /// Related source locations
    pub related: Vec<RelatedInfo>,

    /// How confident we are this is a real error (0.0 - 1.0)
    pub confidence: f32,

    /// Additional notes
    pub notes: Vec<String>,
}

impl RichDiagnostic {
    /// Create a new error diagnostic
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            severity: Severity::Error,
            location: None,
            snippet: None,
            fixes: Vec::new(),
            explanation: None,
            learn_more: None,
            related: Vec::new(),
            confidence: 1.0,
            notes: Vec::new(),
        }
    }

    /// Create a new warning diagnostic
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            severity: Severity::Warning,
            location: None,
            snippet: None,
            fixes: Vec::new(),
            explanation: None,
            learn_more: None,
            related: Vec::new(),
            confidence: 1.0,
            notes: Vec::new(),
        }
    }

    /// Add a source location
    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Add a source snippet
    pub fn with_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.snippet = Some(snippet.into());
        self
    }

    /// Add a fix suggestion
    pub fn with_fix(mut self, fix: Fix) -> Self {
        self.fixes.push(fix);
        self
    }

    /// Add an explanation
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = Some(explanation.into());
        self
    }

    /// Add a note
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Note,
    Help,
}

/// Source location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File path (if available)
    pub file: Option<PathBuf>,
    /// 1-indexed line number
    pub line: usize,
    /// 1-indexed column number
    pub column: usize,
    /// End line (if span covers multiple lines)
    pub end_line: Option<usize>,
    /// End column
    pub end_column: Option<usize>,
}

/// A concrete fix suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fix {
    /// Human-readable description of the fix
    pub description: String,

    /// The actual code change
    pub patch: Patch,

    /// How confident we are this fix is correct (0.0 - 1.0)
    pub confidence: f32,

    /// Does this fix change semantics (vs just fixing syntax/types)?
    pub may_change_semantics: bool,

    /// Category of fix
    pub kind: FixKind,
}

impl Fix {
    /// Create a new fix
    pub fn new(description: impl Into<String>, patch: Patch) -> Self {
        Self {
            description: description.into(),
            patch,
            confidence: 0.8,
            may_change_semantics: false,
            kind: FixKind::Other,
        }
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Mark as potentially changing semantics
    pub fn may_change_semantics(mut self) -> Self {
        self.may_change_semantics = true;
        self
    }

    /// Set the fix kind
    pub fn with_kind(mut self, kind: FixKind) -> Self {
        self.kind = kind;
        self
    }
}

/// The actual text change for a fix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch {
    /// Start line (1-indexed)
    pub start_line: usize,
    /// Start column (1-indexed)
    pub start_column: usize,
    /// End line (1-indexed)
    pub end_line: usize,
    /// End column (1-indexed)
    pub end_column: usize,
    /// Replacement text
    pub replacement: String,
}

impl Patch {
    /// Create a patch that inserts text at a position
    pub fn insert(line: usize, column: usize, text: impl Into<String>) -> Self {
        Self {
            start_line: line,
            start_column: column,
            end_line: line,
            end_column: column,
            replacement: text.into(),
        }
    }

    /// Create a patch that replaces a range
    pub fn replace(
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
        text: impl Into<String>,
    ) -> Self {
        Self {
            start_line,
            start_column,
            end_line,
            end_column,
            replacement: text.into(),
        }
    }

    /// Create a patch that deletes a range
    pub fn delete(
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
    ) -> Self {
        Self {
            start_line,
            start_column,
            end_line,
            end_column,
            replacement: String::new(),
        }
    }
}

/// Category of fix
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixKind {
    /// Add a type cast (e.g., `x as i32`)
    AddCast,
    /// Add a borrow (e.g., `&x`)
    AddBorrow,
    /// Add a mutable borrow (e.g., `&mut x`)
    AddMutBorrow,
    /// Add `.clone()`
    AddClone,
    /// Add `.to_string()` or similar conversion
    AddConversion,
    /// Add a dereference (e.g., `*x`)
    AddDeref,
    /// Add an import statement
    AddImport,
    /// Change a type annotation
    ChangeType,
    /// Remove unused code
    RemoveUnused,
    /// Add lifetime annotation
    AddLifetime,
    /// Other fix type
    Other,
}

/// Information about a related source location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedInfo {
    /// Description of the relationship
    pub message: String,
    /// Location
    pub location: SourceLocation,
}

/// Information about an unknown/unresolved type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnknownInfo {
    /// What we couldn't resolve
    pub description: String,
    /// Where it occurred
    pub location: Option<SourceLocation>,
    /// Why we couldn't resolve it
    pub reason: UnknownReason,
}

/// Reason why something couldn't be resolved
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownReason {
    /// Method not found on type
    MethodNotFound { receiver_type: String, method_name: String },
    /// Field not found on type
    FieldNotFound { type_name: String, field_name: String },
    /// Type not in scope
    TypeNotFound { type_name: String },
    /// Variable not in scope
    VariableNotFound { var_name: String },
    /// External crate not stubbed
    ExternalCrate { crate_name: String },
    /// Generic type couldn't be inferred
    InferenceFailure,
    /// Other reason
    Other { description: String },
}

/// Recommendation for the caller
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recommendation {
    /// Bolt is confident in the result
    TrustBolt,
    /// Some uncertainty - verify critical code with rustc
    VerifyWithRustc,
    /// Too many unknowns - use rustc for this file
    UseRustc { reason: String },
}

/// Timing information for the check
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingInfo {
    /// Total time in milliseconds
    pub total_ms: f64,
    /// Parse time in milliseconds
    pub parse_ms: f64,
    /// Type checking time in milliseconds
    pub typecheck_ms: f64,
    /// Borrow checking time in milliseconds
    pub borrowcheck_ms: f64,
}

/// Statistics about the check
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckStats {
    /// Number of functions checked
    pub functions_checked: usize,
    /// Number of types resolved
    pub types_resolved: usize,
    /// Number of inference variables created
    pub inference_vars: usize,
    /// Number of unknowns encountered
    pub unknowns: usize,
    /// Cache hit rate (if caching enabled)
    pub cache_hit_rate: f64,
}

// ============================================================================
// Parse Result Types
// ============================================================================

/// Result of a parse operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    /// Whether parsing succeeded
    pub success: bool,
    /// Parse errors (if any)
    pub errors: Vec<RichDiagnostic>,
    /// Timing information
    pub parse_time_ms: f64,
}

// ============================================================================
// Core API Functions
// ============================================================================

/// Check Rust source code and return structured diagnostics
///
/// This is the primary entry point for LLM/AI tool integration.
///
/// # Arguments
/// * `source` - Rust source code as a string
/// * `config` - Configuration options
///
/// # Returns
/// A `CheckResult` with errors, warnings, partial types, and fix suggestions
///
/// # Example
/// ```rust,ignore
/// let result = bolt::api::check("fn main() { let x: i32 = \"hello\"; }", CheckConfig::default());
/// assert!(!result.success);
/// assert!(!result.errors.is_empty());
/// ```
pub fn check(source: &str, config: CheckConfig) -> CheckResult {
    check_impl(source, None, config)
}

/// Check Rust source code with a virtual filename
///
/// Same as `check` but allows specifying a filename for error messages.
pub fn check_with_filename(source: &str, filename: &str, config: CheckConfig) -> CheckResult {
    check_impl(source, Some(filename), config)
}

/// Check a file and return structured diagnostics
///
/// # Arguments
/// * `path` - Path to the Rust source file
/// * `config` - Configuration options
///
/// # Returns
/// A `CheckResult` with errors, warnings, partial types, and fix suggestions
pub fn check_file(path: &Path, config: CheckConfig) -> CheckResult {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            return CheckResult::failed(vec![
                RichDiagnostic::error("E0000", format!("Failed to read file: {}", e))
            ]);
        }
    };

    check_impl(&source, path.to_str(), config)
}

/// Parse source code and return parse result
///
/// Useful for tools that need AST access without full type checking.
pub fn parse(source: &str, backend: ParserBackend) -> ParseResult {
    let start = Instant::now();

    // Add to source map
    let file_id = source_map().add_source(PathBuf::from("<input>"), source.to_string());

    let result = match backend {
        ParserBackend::Syn => {
            crate::parser::Parser::new()
                .parse_source(source, "<input>")
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
        ParserBackend::Chumsky => {
            crate::parser2::lower::parse_and_lower_with_file_id(source, "input", file_id)
                .map(|_| ())
                .map_err(|errors| errors.join("\n"))
        }
    };

    let parse_time = start.elapsed();

    match result {
        Ok(()) => ParseResult {
            success: true,
            errors: Vec::new(),
            parse_time_ms: parse_time.as_secs_f64() * 1000.0,
        },
        Err(e) => ParseResult {
            success: false,
            errors: vec![RichDiagnostic::error("E0001", e)],
            parse_time_ms: parse_time.as_secs_f64() * 1000.0,
        },
    }
}

// ============================================================================
// Internal Implementation
// ============================================================================

fn check_impl(source: &str, filename: Option<&str>, config: CheckConfig) -> CheckResult {
    let total_start = Instant::now();
    let filename = filename.unwrap_or("<input>");

    // Add source to source map
    let file_id = source_map().add_source(PathBuf::from(filename), source.to_string());

    // Parse
    let parse_start = Instant::now();
    let krate = match parse_source(source, filename, file_id, config.parser) {
        Ok(k) => k,
        Err(errors) => {
            return CheckResult {
                success: false,
                errors,
                warnings: Vec::new(),
                partial_types: HashMap::new(),
                unknowns: Vec::new(),
                confidence: 1.0,
                recommendation: Recommendation::TrustBolt,
                timing: TimingInfo {
                    total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
                    parse_ms: parse_start.elapsed().as_secs_f64() * 1000.0,
                    typecheck_ms: 0.0,
                    borrowcheck_ms: 0.0,
                },
                stats: CheckStats::default(),
            };
        }
    };
    let parse_time = parse_start.elapsed();

    // Type check
    let typecheck_start = Instant::now();
    let registry = Arc::new(TypeRegistry::new());
    registry.init_from_crate(&krate);

    let type_ctx = TypeContext::new(Arc::clone(&registry));
    let mut type_checker = TypeChecker::new(&type_ctx, &krate);

    if let Err(e) = type_checker.check_crate() {
        return CheckResult::failed(vec![
            RichDiagnostic::error("E0000", e.to_string())
        ]);
    }
    let typecheck_time = typecheck_start.elapsed();

    // Collect type errors
    let mut errors: Vec<RichDiagnostic> = type_ctx
        .take_diagnostics()
        .into_iter()
        .map(|d| diagnostic_to_rich(d, ErrorCode::TypeMismatch, &config))
        .collect();

    // Borrow check
    let borrowcheck_start = Instant::now();
    if config.run_borrow_check {
        let type_alias_map = registry.get_type_alias_map();

        // Fast heuristic checker
        let borrow_checker = BorrowChecker::with_type_aliases(type_alias_map.clone());
        borrow_checker.check_crate(&krate);
        let fast_diags = borrow_checker.take_diagnostics();

        // Full NLL analysis
        let nll_checker = NllChecker::new();
        nll_checker.check_crate(&krate);
        let nll_diags = nll_checker.take_diagnostics();

        // Merge and deduplicate
        let mut seen: HashSet<String> = HashSet::new();
        for d in fast_diags.into_iter().chain(nll_diags.into_iter()) {
            if seen.insert(d.message.clone()) {
                errors.push(diagnostic_to_rich(d, ErrorCode::BorrowOfMovedValue, &config));
            }
        }
    }
    let borrowcheck_time = borrowcheck_start.elapsed();

    // Limit errors if configured
    if config.max_errors > 0 && errors.len() > config.max_errors {
        errors.truncate(config.max_errors);
    }

    // Calculate confidence
    let confidence = calculate_confidence(&errors, &[]);

    // Determine recommendation
    let recommendation = if confidence > 0.9 {
        Recommendation::TrustBolt
    } else if confidence > 0.5 {
        Recommendation::VerifyWithRustc
    } else {
        Recommendation::UseRustc {
            reason: "Too many unknowns or complex patterns".to_string(),
        }
    };

    let total_time = total_start.elapsed();

    CheckResult {
        success: errors.is_empty(),
        errors,
        warnings: Vec::new(),
        partial_types: if config.include_partial_types {
            collect_partial_types(&registry)
        } else {
            HashMap::new()
        },
        unknowns: Vec::new(),
        confidence,
        recommendation,
        timing: TimingInfo {
            total_ms: total_time.as_secs_f64() * 1000.0,
            parse_ms: parse_time.as_secs_f64() * 1000.0,
            typecheck_ms: typecheck_time.as_secs_f64() * 1000.0,
            borrowcheck_ms: borrowcheck_time.as_secs_f64() * 1000.0,
        },
        stats: CheckStats::default(),
    }
}

fn parse_source(
    source: &str,
    filename: &str,
    file_id: u32,
    backend: ParserBackend,
) -> Result<Crate, Vec<RichDiagnostic>> {
    match backend {
        ParserBackend::Syn => {
            crate::parser::Parser::new()
                .parse_source(source, filename)
                .map_err(|e| vec![RichDiagnostic::error("E0001", e.to_string())])
        }
        ParserBackend::Chumsky => {
            crate::parser2::lower::parse_and_lower_with_file_id(source, filename, file_id)
                .map_err(|errors| {
                    errors
                        .into_iter()
                        .map(|e| RichDiagnostic::error("E0001", e))
                        .collect()
                })
        }
    }
}

fn diagnostic_to_rich(d: Diagnostic, default_code: ErrorCode, config: &CheckConfig) -> RichDiagnostic {
    let mut rich = RichDiagnostic::error(
        format!("{:?}", default_code),
        d.message.clone(),
    );

    // Add location if available
    if let Some(span) = d.span {
        if let Some(loc) = source_map().span_to_location(span) {
            rich.location = Some(SourceLocation {
                file: Some(loc.file),
                line: loc.line,
                column: loc.column,
                end_line: loc.end_line,
                end_column: loc.end_column,
            });
        }

        // Add snippet
        if let Some(snippet) = source_map().get_snippet(span) {
            rich.snippet = Some(snippet);
        }
    }

    // Add notes
    for note in d.notes {
        rich.notes.push(note);
    }

    // Generate fix suggestions based on error message patterns
    if config.include_suggestions {
        rich.fixes = generate_fixes(&d.message, d.span);
    }

    // Add explanation if configured
    if config.include_explanations {
        rich.explanation = generate_explanation(&d.message, default_code);
    }

    rich
}

fn generate_fixes(message: &str, span: Option<Span>) -> Vec<Fix> {
    // Use the comprehensive fix generator from the fixes module
    crate::fixes::generate_fixes(message, span, None)
}

fn generate_explanation(message: &str, code: ErrorCode) -> Option<String> {
    match code {
        ErrorCode::TypeMismatch => {
            if message.contains("expected") && message.contains("found") {
                Some("Rust is statically typed. The type you provided doesn't match what was expected. You may need to add a type conversion or change the type annotation.".to_string())
            } else {
                None
            }
        }
        ErrorCode::BorrowOfMovedValue => {
            Some("In Rust, values can only have one owner at a time. When you pass a value to a function or assign it to another variable, ownership moves. You can use .clone() to make a copy, or use references (&) to borrow without moving.".to_string())
        }
        ErrorCode::MutableBorrowConflict => {
            Some("Rust prevents data races by ensuring only one mutable reference exists at a time. You either have multiple mutable references, or a mutable reference while immutable references exist.".to_string())
        }
        _ => None,
    }
}

fn calculate_confidence(errors: &[RichDiagnostic], unknowns: &[UnknownInfo]) -> f32 {
    let base_confidence = 1.0;

    // Reduce confidence based on unknowns
    let unknown_penalty = unknowns.len() as f32 * 0.1;

    // Some error patterns indicate uncertainty
    let uncertain_errors = errors.iter().filter(|e| {
        e.message.contains("unknown") || e.message.contains("could not")
    }).count();
    let uncertainty_penalty = uncertain_errors as f32 * 0.05;

    (base_confidence - unknown_penalty - uncertainty_penalty).max(0.0)
}

fn collect_partial_types(registry: &TypeRegistry) -> HashMap<String, String> {
    // TODO: Extract resolved types from registry
    HashMap::new()
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quick check with lenient settings - optimized for LLM iteration
pub fn quick_check(source: &str) -> CheckResult {
    check(source, CheckConfig::lenient())
}

/// Strict check that matches rustc behavior
pub fn strict_check(source: &str) -> CheckResult {
    check(source, CheckConfig::strict())
}

/// Check and return only the errors as a simple list
pub fn check_errors(source: &str) -> Vec<String> {
    check(source, CheckConfig::default())
        .errors
        .into_iter()
        .map(|e| e.message)
        .collect()
}

/// Check and return JSON result
pub fn check_json(source: &str) -> String {
    check(source, CheckConfig::default()).to_json()
}

/// Check and return pretty JSON result
pub fn check_json_pretty(source: &str) -> String {
    check(source, CheckConfig::default()).to_json_pretty()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_valid_code() {
        let source = "fn main() { let x: i32 = 42; }";
        let result = check(source, CheckConfig::default());
        assert!(result.success);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_check_type_error() {
        let source = r#"fn main() { let x: i32 = "hello"; }"#;
        let result = check(source, CheckConfig::default());
        assert!(!result.success);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_check_config_lenient() {
        let config = CheckConfig::lenient();
        assert_eq!(config.strictness, Strictness::Lenient);
        assert!(config.include_suggestions);
    }

    #[test]
    fn test_check_result_json() {
        let result = CheckResult::success();
        let json = result.to_json();
        assert!(json.contains("success"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_fix_generation() {
        let message = "expected `i32`, found `i64`";
        let fixes = generate_fixes(message, None);
        // Without span, no fixes generated
        assert!(fixes.is_empty());
    }

    #[test]
    fn test_quick_check() {
        let result = quick_check("fn main() {}");
        assert!(result.success);
    }
}
