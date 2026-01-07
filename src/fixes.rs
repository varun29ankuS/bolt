//! Fix Generation for LLM-Optimized Error Recovery
//!
//! This module generates concrete, actionable fix suggestions for Rust errors.
//! Each error type has specialized pattern matching to produce high-confidence fixes.

use crate::error::{source_map, Span};
use crate::api::{Fix, FixKind, Patch};
use std::collections::HashMap;

// ============================================================================
// Fix Generator
// ============================================================================

/// Generates fix suggestions for errors
pub struct FixGenerator {
    /// Known type conversions: (from_type, to_type) -> fix pattern
    type_conversions: HashMap<(String, String), ConversionFix>,
}

/// How to fix a type conversion
#[derive(Clone)]
struct ConversionFix {
    /// What to append/insert
    suffix: String,
    /// Description
    description: String,
    /// Confidence (0.0 - 1.0)
    confidence: f32,
    /// Fix kind
    kind: FixKind,
    /// May change semantics (e.g., truncation)
    may_change_semantics: bool,
}

impl Default for FixGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl FixGenerator {
    pub fn new() -> Self {
        let mut type_conversions = HashMap::new();

        // Integer conversions
        Self::add_int_conversions(&mut type_conversions);

        // String conversions
        Self::add_string_conversions(&mut type_conversions);

        // Reference conversions
        Self::add_ref_conversions(&mut type_conversions);

        // Option/Result conversions
        Self::add_option_result_conversions(&mut type_conversions);

        Self { type_conversions }
    }

    fn add_int_conversions(map: &mut HashMap<(String, String), ConversionFix>) {
        // Widening conversions (safe, no truncation)
        let widening = [
            ("i8", "i16"), ("i8", "i32"), ("i8", "i64"), ("i8", "i128"), ("i8", "isize"),
            ("i16", "i32"), ("i16", "i64"), ("i16", "i128"), ("i16", "isize"),
            ("i32", "i64"), ("i32", "i128"),
            ("i64", "i128"),
            ("u8", "u16"), ("u8", "u32"), ("u8", "u64"), ("u8", "u128"), ("u8", "usize"),
            ("u8", "i16"), ("u8", "i32"), ("u8", "i64"), ("u8", "i128"),
            ("u16", "u32"), ("u16", "u64"), ("u16", "u128"), ("u16", "usize"),
            ("u16", "i32"), ("u16", "i64"), ("u16", "i128"),
            ("u32", "u64"), ("u32", "u128"),
            ("u32", "i64"), ("u32", "i128"),
            ("u64", "u128"), ("u64", "i128"),
        ];

        for (from, to) in widening {
            map.insert(
                (from.to_string(), to.to_string()),
                ConversionFix {
                    suffix: format!(" as {}", to),
                    description: format!("Cast {} to {} (widening, safe)", from, to),
                    confidence: 0.95,
                    kind: FixKind::AddCast,
                    may_change_semantics: false,
                },
            );
        }

        // Narrowing conversions (may truncate)
        let narrowing = [
            ("i128", "i64"), ("i128", "i32"), ("i128", "i16"), ("i128", "i8"),
            ("i64", "i32"), ("i64", "i16"), ("i64", "i8"), ("i64", "isize"),
            ("i32", "i16"), ("i32", "i8"), ("i32", "isize"),
            ("i16", "i8"),
            ("u128", "u64"), ("u128", "u32"), ("u128", "u16"), ("u128", "u8"),
            ("u64", "u32"), ("u64", "u16"), ("u64", "u8"), ("u64", "usize"),
            ("u32", "u16"), ("u32", "u8"), ("u32", "usize"),
            ("u16", "u8"),
            ("usize", "u32"), ("usize", "u16"), ("usize", "u8"),
            ("isize", "i32"), ("isize", "i16"), ("isize", "i8"),
        ];

        for (from, to) in narrowing {
            map.insert(
                (from.to_string(), to.to_string()),
                ConversionFix {
                    suffix: format!(" as {}", to),
                    description: format!("Cast {} to {} (may truncate)", from, to),
                    confidence: 0.85,
                    kind: FixKind::AddCast,
                    may_change_semantics: true,
                },
            );
        }

        // Signed/unsigned conversions
        let sign_changes = [
            ("i8", "u8"), ("u8", "i8"),
            ("i16", "u16"), ("u16", "i16"),
            ("i32", "u32"), ("u32", "i32"),
            ("i64", "u64"), ("u64", "i64"),
            ("i128", "u128"), ("u128", "i128"),
            ("isize", "usize"), ("usize", "isize"),
        ];

        for (from, to) in sign_changes {
            map.insert(
                (from.to_string(), to.to_string()),
                ConversionFix {
                    suffix: format!(" as {}", to),
                    description: format!("Cast {} to {} (sign change, verify range)", from, to),
                    confidence: 0.80,
                    kind: FixKind::AddCast,
                    may_change_semantics: true,
                },
            );
        }

        // Float conversions
        map.insert(
            ("f32".to_string(), "f64".to_string()),
            ConversionFix {
                suffix: " as f64".to_string(),
                description: "Cast f32 to f64 (widening, safe)".to_string(),
                confidence: 0.95,
                kind: FixKind::AddCast,
                may_change_semantics: false,
            },
        );
        map.insert(
            ("f64".to_string(), "f32".to_string()),
            ConversionFix {
                suffix: " as f32".to_string(),
                description: "Cast f64 to f32 (may lose precision)".to_string(),
                confidence: 0.85,
                kind: FixKind::AddCast,
                may_change_semantics: true,
            },
        );
    }

    fn add_string_conversions(map: &mut HashMap<(String, String), ConversionFix>) {
        // String <-> &str
        map.insert(
            ("String".to_string(), "&str".to_string()),
            ConversionFix {
                suffix: ".as_str()".to_string(),
                description: "Convert String to &str with .as_str()".to_string(),
                confidence: 0.95,
                kind: FixKind::AddConversion,
                may_change_semantics: false,
            },
        );
        map.insert(
            ("&str".to_string(), "String".to_string()),
            ConversionFix {
                suffix: ".to_string()".to_string(),
                description: "Convert &str to String with .to_string()".to_string(),
                confidence: 0.95,
                kind: FixKind::AddConversion,
                may_change_semantics: false,
            },
        );
        map.insert(
            ("&String".to_string(), "&str".to_string()),
            ConversionFix {
                suffix: ".as_str()".to_string(),
                description: "Convert &String to &str with .as_str()".to_string(),
                confidence: 0.95,
                kind: FixKind::AddConversion,
                may_change_semantics: false,
            },
        );

        // char <-> String
        map.insert(
            ("char".to_string(), "String".to_string()),
            ConversionFix {
                suffix: ".to_string()".to_string(),
                description: "Convert char to String with .to_string()".to_string(),
                confidence: 0.95,
                kind: FixKind::AddConversion,
                may_change_semantics: false,
            },
        );
    }

    fn add_ref_conversions(map: &mut HashMap<(String, String), ConversionFix>) {
        // These are handled specially in generate_fixes since they need prefix changes
    }

    fn add_option_result_conversions(map: &mut HashMap<(String, String), ConversionFix>) {
        // Option<T> -> T
        map.insert(
            ("Option<T>".to_string(), "T".to_string()),
            ConversionFix {
                suffix: ".unwrap()".to_string(),
                description: "Unwrap Option (panics if None)".to_string(),
                confidence: 0.60,
                kind: FixKind::AddConversion,
                may_change_semantics: true,
            },
        );
        // Result<T, E> -> T
        map.insert(
            ("Result<T, E>".to_string(), "T".to_string()),
            ConversionFix {
                suffix: ".unwrap()".to_string(),
                description: "Unwrap Result (panics if Err)".to_string(),
                confidence: 0.60,
                kind: FixKind::AddConversion,
                may_change_semantics: true,
            },
        );
    }

    /// Generate fixes for an error
    pub fn generate_fixes(&self, message: &str, span: Option<Span>, code: Option<&str>) -> Vec<Fix> {
        let mut fixes = Vec::new();

        // Try all fix generators
        fixes.extend(self.fix_type_mismatch(message, span));
        fixes.extend(self.fix_borrow_error(message, span));
        fixes.extend(self.fix_move_error(message, span));
        fixes.extend(self.fix_missing_trait(message, span));
        fixes.extend(self.fix_unused_variable(message, span));
        fixes.extend(self.fix_missing_field(message, span));
        fixes.extend(self.fix_unknown_method(message, span));

        // Sort by confidence (highest first)
        fixes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        fixes
    }

    /// Fix type mismatch errors
    fn fix_type_mismatch(&self, message: &str, span: Option<Span>) -> Vec<Fix> {
        let mut fixes = Vec::new();

        // Pattern: "expected `X`, found `Y`"
        if let Some((expected, found)) = Self::parse_type_mismatch(message) {
            let loc = span.and_then(|s| source_map().span_to_location(s));

            // Look up conversion in our table
            if let Some(conv) = self.type_conversions.get(&(found.clone(), expected.clone())) {
                if let Some(ref loc) = loc {
                    fixes.push(Fix {
                        description: conv.description.clone(),
                        patch: Patch::insert(
                            loc.line,
                            loc.end_column.unwrap_or(loc.column),
                            &conv.suffix,
                        ),
                        confidence: conv.confidence,
                        may_change_semantics: conv.may_change_semantics,
                        kind: conv.kind,
                    });
                }
            }

            // Special case: need reference
            if expected.starts_with('&') && !found.starts_with('&') {
                let inner = expected.trim_start_matches('&').trim_start_matches("mut ");
                if inner == found || inner == found.trim_start_matches('&') {
                    if let Some(ref loc) = loc {
                        let prefix = if expected.contains("mut") { "&mut " } else { "&" };
                        fixes.push(Fix {
                            description: format!("Add {} borrow", if expected.contains("mut") { "mutable" } else { "immutable" }),
                            patch: Patch::insert(loc.line, loc.column, prefix),
                            confidence: 0.90,
                            may_change_semantics: false,
                            kind: if expected.contains("mut") { FixKind::AddMutBorrow } else { FixKind::AddBorrow },
                        });
                    }
                }
            }

            // Special case: need dereference
            if !expected.starts_with('&') && found.starts_with('&') {
                let inner = found.trim_start_matches('&').trim_start_matches("mut ");
                if inner == expected {
                    if let Some(ref loc) = loc {
                        fixes.push(Fix {
                            description: "Add dereference".to_string(),
                            patch: Patch::insert(loc.line, loc.column, "*"),
                            confidence: 0.90,
                            may_change_semantics: false,
                            kind: FixKind::AddDeref,
                        });
                    }
                }
            }

            // Special case: slice from array
            if expected == "&[T]" || expected.starts_with("&[") {
                if found.starts_with('[') && found.contains(';') {
                    if let Some(ref loc) = loc {
                        fixes.push(Fix {
                            description: "Borrow as slice with &".to_string(),
                            patch: Patch::insert(loc.line, loc.column, "&"),
                            confidence: 0.92,
                            may_change_semantics: false,
                            kind: FixKind::AddBorrow,
                        });
                    }
                }
            }
        }

        fixes
    }

    /// Fix borrow checker errors
    fn fix_borrow_error(&self, message: &str, span: Option<Span>) -> Vec<Fix> {
        let mut fixes = Vec::new();
        let loc = span.and_then(|s| source_map().span_to_location(s));

        // Pattern: "cannot borrow `x` as mutable because it is also borrowed as immutable"
        if message.contains("cannot borrow") && message.contains("as mutable") {
            if let Some(ref loc) = loc {
                fixes.push(Fix {
                    description: "Clone the value to avoid borrow conflict".to_string(),
                    patch: Patch::insert(loc.line, loc.end_column.unwrap_or(loc.column), ".clone()"),
                    confidence: 0.70,
                    may_change_semantics: false,
                    kind: FixKind::AddClone,
                });
            }
        }

        // Pattern: "cannot borrow `x` as mutable more than once"
        if message.contains("cannot borrow") && message.contains("more than once") {
            if let Some(ref loc) = loc {
                fixes.push(Fix {
                    description: "Clone to get independent mutable access".to_string(),
                    patch: Patch::insert(loc.line, loc.end_column.unwrap_or(loc.column), ".clone()"),
                    confidence: 0.65,
                    may_change_semantics: true,
                    kind: FixKind::AddClone,
                });
            }
        }

        fixes
    }

    /// Fix move errors
    fn fix_move_error(&self, message: &str, span: Option<Span>) -> Vec<Fix> {
        let mut fixes = Vec::new();
        let loc = span.and_then(|s| source_map().span_to_location(s));

        // Pattern: "use of moved value" or "value used after move"
        if message.contains("moved value") || message.contains("after move") || message.contains("value borrowed here after move") {
            if let Some(ref loc) = loc {
                // Suggest clone at the original use
                fixes.push(Fix {
                    description: "Clone the value before the move".to_string(),
                    patch: Patch::insert(loc.line, loc.end_column.unwrap_or(loc.column), ".clone()"),
                    confidence: 0.75,
                    may_change_semantics: false,
                    kind: FixKind::AddClone,
                });

                // Suggest borrowing instead
                fixes.push(Fix {
                    description: "Borrow instead of moving".to_string(),
                    patch: Patch::insert(loc.line, loc.column, "&"),
                    confidence: 0.70,
                    may_change_semantics: true,
                    kind: FixKind::AddBorrow,
                });
            }
        }

        // Pattern: "cannot move out of `x` which is behind a shared reference"
        if message.contains("cannot move out of") && message.contains("shared reference") {
            if let Some(ref loc) = loc {
                fixes.push(Fix {
                    description: "Clone to get owned value from reference".to_string(),
                    patch: Patch::insert(loc.line, loc.end_column.unwrap_or(loc.column), ".clone()"),
                    confidence: 0.85,
                    may_change_semantics: false,
                    kind: FixKind::AddClone,
                });
            }
        }

        fixes
    }

    /// Fix missing trait implementation errors
    fn fix_missing_trait(&self, message: &str, span: Option<Span>) -> Vec<Fix> {
        let mut fixes = Vec::new();
        let loc = span.and_then(|s| source_map().span_to_location(s));

        // Pattern: "the trait `X` is not implemented for `Y`"
        if message.contains("trait") && message.contains("is not implemented") {
            // Extract trait name
            if let Some(trait_name) = Self::extract_between(message, "trait `", "` is not") {
                match trait_name.as_str() {
                    "Clone" => {
                        // Suggest deriving Clone
                        fixes.push(Fix {
                            description: "Add #[derive(Clone)] to the type definition".to_string(),
                            patch: Patch::insert(1, 1, "#[derive(Clone)]\n"),
                            confidence: 0.60,
                            may_change_semantics: false,
                            kind: FixKind::Other,
                        });
                    }
                    "Debug" => {
                        fixes.push(Fix {
                            description: "Add #[derive(Debug)] to the type definition".to_string(),
                            patch: Patch::insert(1, 1, "#[derive(Debug)]\n"),
                            confidence: 0.60,
                            may_change_semantics: false,
                            kind: FixKind::Other,
                        });
                    }
                    "Display" => {
                        if let Some(ref loc) = loc {
                            fixes.push(Fix {
                                description: "Use Debug formatting with {:?}".to_string(),
                                patch: Patch::replace(loc.line, loc.column, loc.line, loc.end_column.unwrap_or(loc.column), "{:?}"),
                                confidence: 0.55,
                                may_change_semantics: true,
                                kind: FixKind::Other,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        fixes
    }

    /// Fix unused variable warnings
    fn fix_unused_variable(&self, message: &str, span: Option<Span>) -> Vec<Fix> {
        let mut fixes = Vec::new();
        let loc = span.and_then(|s| source_map().span_to_location(s));

        // Pattern: "unused variable: `x`"
        if message.contains("unused variable") {
            if let Some(var_name) = Self::extract_between(message, "`", "`") {
                if let Some(ref loc) = loc {
                    fixes.push(Fix {
                        description: format!("Prefix with underscore: _{}", var_name),
                        patch: Patch::replace(
                            loc.line, loc.column,
                            loc.line, loc.column + var_name.len(),
                            &format!("_{}", var_name),
                        ),
                        confidence: 0.95,
                        may_change_semantics: false,
                        kind: FixKind::RemoveUnused,
                    });
                }
            }
        }

        fixes
    }

    /// Fix missing field errors
    fn fix_missing_field(&self, message: &str, span: Option<Span>) -> Vec<Fix> {
        let mut fixes = Vec::new();

        // Pattern: "missing field `x` in initializer"
        if message.contains("missing field") {
            if let Some(field_name) = Self::extract_between(message, "field `", "`") {
                if let Some(span) = span {
                    if let Some(loc) = source_map().span_to_location(span) {
                        fixes.push(Fix {
                            description: format!("Add missing field: {}", field_name),
                            patch: Patch::insert(
                                loc.line,
                                loc.end_column.unwrap_or(loc.column),
                                &format!("\n    {}: todo!(\"fill in {}\"),", field_name, field_name),
                            ),
                            confidence: 0.80,
                            may_change_semantics: false,
                            kind: FixKind::Other,
                        });
                    }
                }
            }
        }

        fixes
    }

    /// Fix unknown method errors
    fn fix_unknown_method(&self, message: &str, span: Option<Span>) -> Vec<Fix> {
        let mut fixes = Vec::new();

        // Pattern: "no method named `x` found"
        if message.contains("no method named") {
            // Common typos and alternatives
            let method_fixes: &[(&str, &str, &str)] = &[
                ("len", "length", "Use .len() instead of .length()"),
                ("length", "len", "Use .len() instead of .length()"),
                ("size", "len", "Use .len() instead of .size()"),
                ("count", "len", "Use .len() instead of .count() (for collections)"),
                ("push_back", "push", "Use .push() instead of .push_back()"),
                ("pop_back", "pop", "Use .pop() instead of .pop_back()"),
                ("append", "push", "Use .push() for single elements"),
                ("empty", "is_empty", "Use .is_empty() instead of .empty()"),
            ];

            for (wrong, correct, desc) in method_fixes {
                if message.contains(&format!("`{}`", wrong)) {
                    if let Some(span) = span {
                        if let Some(loc) = source_map().span_to_location(span) {
                            fixes.push(Fix {
                                description: desc.to_string(),
                                patch: Patch::replace(
                                    loc.line, loc.column,
                                    loc.line, loc.column + wrong.len(),
                                    *correct,
                                ),
                                confidence: 0.85,
                                may_change_semantics: false,
                                kind: FixKind::Other,
                            });
                        }
                    }
                }
            }
        }

        fixes
    }

    // ========================================================================
    // Helper functions
    // ========================================================================

    /// Parse "expected `X`, found `Y`" pattern
    fn parse_type_mismatch(message: &str) -> Option<(String, String)> {
        // Try various patterns
        let patterns = [
            ("expected `", "`, found `", "`"),
            ("expected ", ", found ", ""),
            ("expected type `", "`, found `", "`"),
        ];

        for (prefix, middle, suffix) in patterns {
            if let Some(start) = message.find(prefix) {
                let after_prefix = &message[start + prefix.len()..];
                if let Some(mid) = after_prefix.find(middle) {
                    let expected = &after_prefix[..mid];
                    let after_middle = &after_prefix[mid + middle.len()..];

                    let found = if suffix.is_empty() {
                        after_middle.split_whitespace().next().unwrap_or("")
                    } else if let Some(end) = after_middle.find(suffix) {
                        &after_middle[..end]
                    } else {
                        continue;
                    };

                    return Some((
                        expected.trim_matches('`').to_string(),
                        found.trim_matches('`').to_string(),
                    ));
                }
            }
        }

        None
    }

    /// Extract text between two delimiters
    fn extract_between(text: &str, start: &str, end: &str) -> Option<String> {
        let start_idx = text.find(start)? + start.len();
        let remaining = &text[start_idx..];
        let end_idx = remaining.find(end)?;
        Some(remaining[..end_idx].to_string())
    }
}

// ============================================================================
// Global instance
// ============================================================================

static FIX_GENERATOR: once_cell::sync::Lazy<FixGenerator> =
    once_cell::sync::Lazy::new(FixGenerator::new);

/// Get the global fix generator
pub fn fix_generator() -> &'static FixGenerator {
    &FIX_GENERATOR
}

/// Convenience function to generate fixes
pub fn generate_fixes(message: &str, span: Option<Span>, code: Option<&str>) -> Vec<Fix> {
    fix_generator().generate_fixes(message, span, code)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_type_mismatch() {
        let msg = "expected `i32`, found `i64`";
        let result = FixGenerator::parse_type_mismatch(msg);
        assert_eq!(result, Some(("i32".to_string(), "i64".to_string())));
    }

    #[test]
    fn test_parse_type_mismatch_string() {
        let msg = "expected `&str`, found `String`";
        let result = FixGenerator::parse_type_mismatch(msg);
        assert_eq!(result, Some(("&str".to_string(), "String".to_string())));
    }

    #[test]
    fn test_extract_between() {
        let msg = "unused variable: `foo`";
        let result = FixGenerator::extract_between(msg, "`", "`");
        assert_eq!(result, Some("foo".to_string()));
    }

    #[test]
    fn test_fix_generator_has_conversions() {
        let gen = FixGenerator::new();
        assert!(gen.type_conversions.contains_key(&("i64".to_string(), "i32".to_string())));
        assert!(gen.type_conversions.contains_key(&("String".to_string(), "&str".to_string())));
    }
}
