//! Async Borrow Checker
//!
//! Runs borrow checking in the background while code executes immediately.
//! Implements the "compile first, verify later" philosophy.

use super::BorrowChecker;
use crate::error::Diagnostic;
use crate::hir::Crate;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

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
    /// Per-file check state
    file_states: RwLock<HashMap<PathBuf, FileCheckState>>,
}

impl AsyncBorrowChecker {
    pub fn new() -> Self {
        Self {
            file_states: RwLock::new(HashMap::new()),
        }
    }

    /// Check if a previous borrow check failed for this file.
    /// If so, block and return the errors.
    /// Call this BEFORE starting compilation.
    pub fn check_previous_failure(&self, path: &PathBuf) -> Option<Vec<Diagnostic>> {
        let mut states = self.file_states.write();

        if let Some(state) = states.get_mut(path) {
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
                        // Thread panicked, treat as success (don't block)
                        state.result = CheckResult::Success;
                    }
                }
            }

            // Check if previous result was a failure
            match &state.result {
                CheckResult::Failed(diagnostics) => {
                    let diags = diagnostics.clone();
                    // Clear the failure so we can try again
                    state.result = CheckResult::Pending;
                    Some(diags)
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Spawn borrow checking in the background.
    /// Returns immediately - check result will be available later.
    pub fn spawn_check(&self, path: PathBuf, krate: Arc<Crate>) {
        // Clone what we need for the thread
        let path_clone = path.clone();

        let handle = thread::spawn(move || {
            let checker = BorrowChecker::new();
            checker.check_crate(&krate);
            checker.take_diagnostics()
        });

        let mut states = self.file_states.write();
        states.insert(
            path,
            FileCheckState {
                result: CheckResult::Pending,
                handle: Some(handle),
            },
        );
    }

    /// Wait for the current check to complete and get results.
    /// Call this AFTER execution to show any warnings.
    pub fn wait_for_result(&self, path: &PathBuf) -> CheckResult {
        let mut states = self.file_states.write();

        if let Some(state) = states.get_mut(path) {
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
            state.result.clone()
        } else {
            CheckResult::Success
        }
    }

    /// Try to get result without blocking.
    /// Returns None if check is still pending.
    pub fn try_get_result(&self, path: &PathBuf) -> Option<CheckResult> {
        let mut states = self.file_states.write();

        if let Some(state) = states.get_mut(path) {
            // Check if thread is done without blocking
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
                    // Still running, put handle back
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
        self.file_states.write().remove(path);
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
