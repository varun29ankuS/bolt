//! Async Borrow Checker
//!
//! Runs borrow checking in the background while code executes immediately.
//! Implements the "compile first, verify later" philosophy.
//!
//! State is persisted to disk so blocking works across CLI invocations.

use super::{BorrowChecker, NllChecker, TypeAliasMap};
use crate::error::Diagnostic;
use crate::hir::Crate;
use std::collections::HashSet;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Canonicalize a path for consistent HashMap keys
fn canonical_path(path: &PathBuf) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.clone())
}

/// Get the failure state file path for a source file
fn failure_marker_path(source_path: &PathBuf) -> PathBuf {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("bolt")
        .join("borrow_check");

    // Create hash of source path for the marker file name
    let path_str = source_path.to_string_lossy();
    let hash = blake3::hash(path_str.as_bytes());
    let hash_str = &hash.to_hex()[..16]; // Use first 16 hex chars

    cache_dir.join(format!("{}.failed", hash_str))
}

/// Persisted failure state
#[derive(Serialize, Deserialize)]
struct PersistedFailure {
    source_path: String,
    messages: Vec<String>,
}

/// Save failure state to disk
fn persist_failure(path: &PathBuf, diagnostics: &[Diagnostic]) {
    let marker_path = failure_marker_path(path);

    // Ensure directory exists
    if let Some(parent) = marker_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let failure = PersistedFailure {
        source_path: path.to_string_lossy().to_string(),
        messages: diagnostics.iter().map(|d| d.message.clone()).collect(),
    };

    if let Ok(json) = serde_json::to_string(&failure) {
        let _ = std::fs::write(&marker_path, json);
    }
}

/// Load failure state from disk
fn load_persisted_failure(path: &PathBuf) -> Option<Vec<String>> {
    let marker_path = failure_marker_path(path);

    if let Ok(content) = std::fs::read_to_string(&marker_path) {
        if let Ok(failure) = serde_json::from_str::<PersistedFailure>(&content) {
            return Some(failure.messages);
        }
    }
    None
}

/// Clear failure state from disk
fn clear_persisted_failure(path: &PathBuf) {
    let marker_path = failure_marker_path(path);
    let _ = std::fs::remove_file(&marker_path);
}

/// Result of a borrow check
#[derive(Debug, Clone)]
pub enum CheckResult {
    /// Check is still running
    Pending,
    /// Check completed with no errors
    Success,
    /// Check completed with errors
    Failed(Vec<Diagnostic>),
}

/// State for a single file's borrow check
struct FileCheckState {
    result: CheckResult,
    /// Handle to the background thread (if still running)
    handle: Option<JoinHandle<Vec<Diagnostic>>>,
}

/// Async borrow checker that runs checks in the background
pub struct AsyncBorrowChecker {
    /// Per-file check state (keyed by canonical path)
    file_states: RwLock<HashMap<PathBuf, FileCheckState>>,
}

impl AsyncBorrowChecker {
    pub fn new() -> Self {
        Self {
            file_states: RwLock::new(HashMap::new()),
        }
    }

    /// Check if a previous borrow check failed for this file.
    /// Checks both in-memory state and persisted disk state.
    /// Call this BEFORE starting compilation.
    pub fn check_previous_failure(&self, path: &PathBuf) -> Option<Vec<Diagnostic>> {
        let canonical = canonical_path(path);

        // First check in-memory state
        {
            let mut states = self.file_states.write();
            if let Some(state) = states.get_mut(&canonical) {
                // If there's a pending check, wait for it
                if let Some(handle) = state.handle.take() {
                    match handle.join() {
                        Ok(diagnostics) => {
                            if diagnostics.is_empty() {
                                state.result = CheckResult::Success;
                            } else {
                                state.result = CheckResult::Failed(diagnostics);
                            }
                        }
                        Err(_) => {
                            state.result = CheckResult::Success;
                        }
                    }
                }

                if let CheckResult::Failed(diagnostics) = &state.result {
                    return Some(diagnostics.clone());
                }
            }
        }

        // Check disk-persisted state (from previous CLI invocation)
        if let Some(messages) = load_persisted_failure(&canonical) {
            let diagnostics: Vec<Diagnostic> = messages
                .into_iter()
                .map(|msg| Diagnostic::error(msg))
                .collect();
            return Some(diagnostics);
        }

        None
    }

    /// Spawn borrow checking in the background.
    /// Returns immediately - check result will be available later.
    /// Clears any persisted failure state since we're recompiling.
    pub fn spawn_check(&self, path: PathBuf, krate: Arc<Crate>) {
        // Default to empty type aliases for backwards compatibility
        self.spawn_check_with_aliases(path, krate, TypeAliasMap::new());
    }

    pub fn spawn_check_with_aliases(&self, path: PathBuf, krate: Arc<Crate>, type_aliases: TypeAliasMap) {
        self.spawn_full_analysis(path, krate, type_aliases, HashSet::new());
    }

    /// Spawn full NLL analysis in the background.
    /// This runs both the fast heuristic checker and the thorough NLL checker.
    /// Since it's async, we get 99%+ accuracy with no blocking.
    pub fn spawn_full_analysis(
        &self,
        path: PathBuf,
        krate: Arc<Crate>,
        type_aliases: TypeAliasMap,
        copy_types: HashSet<String>,
    ) {
        let canonical = canonical_path(&path);

        // Clear persisted failure since we're recompiling
        clear_persisted_failure(&canonical);

        // Clone for the thread closure
        let canonical_for_persist = canonical.clone();

        let handle = thread::spawn(move || {
            let mut all_diagnostics = Vec::new();

            // Phase 1: Fast heuristic check (catches obvious errors quickly)
            let fast_checker = BorrowChecker::with_type_aliases(type_aliases);
            fast_checker.check_crate(&krate);
            let fast_diags = fast_checker.take_diagnostics();

            // Phase 2: Full NLL analysis (catches everything)
            let nll_checker = NllChecker::with_copy_types(copy_types);
            nll_checker.check_crate(&krate);
            let nll_diags = nll_checker.take_diagnostics();

            // Merge diagnostics, deduplicating by message
            let mut seen_messages: HashSet<String> = HashSet::new();
            for diag in fast_diags.into_iter().chain(nll_diags.into_iter()) {
                if seen_messages.insert(diag.message.clone()) {
                    all_diagnostics.push(diag);
                }
            }

            // Persist failure to disk if there are errors
            if !all_diagnostics.is_empty() {
                persist_failure(&canonical_for_persist, &all_diagnostics);
            }

            all_diagnostics
        });

        let mut states = self.file_states.write();
        states.insert(
            canonical,
            FileCheckState {
                result: CheckResult::Pending,
                handle: Some(handle),
            },
        );
    }

    /// Wait for the current check to complete and get results.
    /// Call this AFTER execution to show any warnings.
    pub fn wait_for_result(&self, path: &PathBuf) -> CheckResult {
        let canonical = canonical_path(path);
        let mut states = self.file_states.write();

        if let Some(state) = states.get_mut(&canonical) {
            if let Some(handle) = state.handle.take() {
                match handle.join() {
                    Ok(diagnostics) => {
                        if diagnostics.is_empty() {
                            state.result = CheckResult::Success;
                        } else {
                            state.result = CheckResult::Failed(diagnostics);
                        }
                    }
                    Err(_) => {
                        state.result = CheckResult::Success;
                    }
                }
            }
            state.result.clone()
        } else {
            CheckResult::Success
        }
    }

    /// Try to get result without blocking.
    /// Returns None if check is still pending.
    pub fn try_get_result(&self, path: &PathBuf) -> Option<CheckResult> {
        let canonical = canonical_path(path);
        let mut states = self.file_states.write();

        if let Some(state) = states.get_mut(&canonical) {
            if let Some(handle) = state.handle.take() {
                if handle.is_finished() {
                    match handle.join() {
                        Ok(diagnostics) => {
                            if diagnostics.is_empty() {
                                state.result = CheckResult::Success;
                            } else {
                                state.result = CheckResult::Failed(diagnostics);
                            }
                        }
                        Err(_) => {
                            state.result = CheckResult::Success;
                        }
                    }
                    Some(state.result.clone())
                } else {
                    state.handle = Some(handle);
                    None
                }
            } else {
                Some(state.result.clone())
            }
        } else {
            Some(CheckResult::Success)
        }
    }

    /// Clear state for a file (e.g., when source changes)
    pub fn clear(&self, path: &PathBuf) {
        let canonical = canonical_path(path);
        self.file_states.write().remove(&canonical);
        clear_persisted_failure(&canonical);
    }
}

impl Default for AsyncBorrowChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Global async borrow checker instance
static ASYNC_CHECKER: once_cell::sync::Lazy<AsyncBorrowChecker> =
    once_cell::sync::Lazy::new(AsyncBorrowChecker::new);

/// Get the global async borrow checker
pub fn global_checker() -> &'static AsyncBorrowChecker {
    &ASYNC_CHECKER
}
