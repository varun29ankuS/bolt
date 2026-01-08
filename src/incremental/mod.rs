//! Incremental Compilation Support
//!
//! Tracks file dependencies and only rechecks what changed.
//!
//! # Architecture
//!
//! ```text
//! File Change → Hash Check → Dependency Graph → Dirty Set → Partial Recheck
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use blake3::Hash;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::hir::{Crate, DefId, Item};

/// File state for incremental compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    /// Content hash
    pub hash: String,
    /// Last modified time
    pub mtime: u64,
    /// Items defined in this file
    pub defines: Vec<DefId>,
    /// Items this file depends on
    pub depends_on: Vec<DefId>,
    /// Other files this file imports (mod statements)
    pub imports: Vec<PathBuf>,
}

/// Dependency graph for incremental compilation
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// File path -> File state
    files: DashMap<PathBuf, FileState>,
    /// DefId -> File that defines it
    def_to_file: DashMap<DefId, PathBuf>,
    /// Reverse dependencies: DefId -> Files that depend on it
    dependents: DashMap<DefId, HashSet<PathBuf>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a file has changed since last check
    pub fn has_changed(&self, path: &Path) -> bool {
        let current_hash = match Self::hash_file(path) {
            Some(h) => h,
            None => return true, // Can't read = assume changed
        };

        match self.files.get(path) {
            Some(state) => state.hash != current_hash,
            None => true, // New file
        }
    }

    /// Hash a file's contents
    pub fn hash_file(path: &Path) -> Option<String> {
        let content = std::fs::read(path).ok()?;
        Some(blake3::hash(&content).to_hex().to_string())
    }

    /// Get mtime of a file
    fn get_mtime(path: &Path) -> u64 {
        std::fs::metadata(path)
            .and_then(|m| m.modified())
            .map(|t| t.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs())
            .unwrap_or(0)
    }

    /// Update file state after parsing
    pub fn update_file(&self, path: PathBuf, krate: &Crate) {
        let hash = Self::hash_file(&path).unwrap_or_default();
        let mtime = Self::get_mtime(&path);

        // Collect definitions from this file
        let defines: Vec<DefId> = krate.items.keys().cloned().collect();

        // Track which file defines each item
        for def_id in &defines {
            self.def_to_file.insert(*def_id, path.clone());
        }

        // TODO: Extract dependencies from type references
        let depends_on = Vec::new();
        let imports = Vec::new();

        self.files.insert(path, FileState {
            hash,
            mtime,
            defines,
            depends_on,
            imports,
        });
    }

    /// Get all files that need rechecking given a changed file
    pub fn get_dirty_files(&self, changed: &Path) -> HashSet<PathBuf> {
        let mut dirty = HashSet::new();
        dirty.insert(changed.to_path_buf());

        // Get items defined in the changed file
        if let Some(state) = self.files.get(changed) {
            for def_id in &state.defines {
                // Find all files that depend on these items
                if let Some(deps) = self.dependents.get(def_id) {
                    dirty.extend(deps.iter().cloned());
                }
            }
        }

        // Recursively get transitive dependents
        let mut to_check: Vec<PathBuf> = dirty.iter().cloned().collect();
        while let Some(path) = to_check.pop() {
            if let Some(state) = self.files.get(&path) {
                for def_id in &state.defines {
                    if let Some(deps) = self.dependents.get(def_id) {
                        for dep in deps.iter() {
                            if dirty.insert(dep.clone()) {
                                to_check.push(dep.clone());
                            }
                        }
                    }
                }
            }
        }

        dirty
    }

    /// Register a dependency: `dependent` depends on `dependency`
    pub fn add_dependency(&self, dependent: PathBuf, dependency: DefId) {
        self.dependents
            .entry(dependency)
            .or_insert_with(HashSet::new)
            .insert(dependent);
    }

    /// Clear all cached state
    pub fn clear(&self) {
        self.files.clear();
        self.def_to_file.clear();
        self.dependents.clear();
    }

    /// Save dependency graph to disk
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let data: HashMap<PathBuf, FileState> = self.files
            .iter()
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect();

        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load dependency graph from disk
    pub fn load(&self, path: &Path) -> std::io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let data: HashMap<PathBuf, FileState> = serde_json::from_str(&json)?;

        for (path, state) in data {
            // Rebuild def_to_file mapping
            for def_id in &state.defines {
                self.def_to_file.insert(*def_id, path.clone());
            }
            self.files.insert(path, state);
        }

        Ok(())
    }
}

/// Incremental checker that only rechecks changed files
pub struct IncrementalChecker {
    /// Dependency graph
    pub graph: DependencyGraph,
    /// Cache directory
    cache_dir: PathBuf,
}

impl IncrementalChecker {
    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("bolt")
            .join("incremental");

        std::fs::create_dir_all(&cache_dir).ok();

        let checker = Self {
            graph: DependencyGraph::new(),
            cache_dir,
        };

        // Try to load existing graph
        let graph_path = checker.cache_dir.join("deps.json");
        checker.graph.load(&graph_path).ok();

        checker
    }

    /// Check which files need rechecking in a project
    pub fn get_files_to_check(&self, root: &Path) -> Vec<PathBuf> {
        let mut to_check = Vec::new();

        // Walk all .rs files
        if let Ok(entries) = walkdir(root) {
            for entry in entries {
                if entry.extension().map(|e| e == "rs").unwrap_or(false) {
                    if self.graph.has_changed(&entry) {
                        to_check.push(entry);
                    }
                }
            }
        }

        // Expand to include dependents
        let mut dirty = HashSet::new();
        for path in &to_check {
            dirty.extend(self.graph.get_dirty_files(path));
        }

        dirty.into_iter().collect()
    }

    /// Update cache after successful check
    pub fn update_cache(&self, path: &Path, krate: &Crate) {
        self.graph.update_file(path.to_path_buf(), krate);

        // Persist to disk
        let graph_path = self.cache_dir.join("deps.json");
        self.graph.save(&graph_path).ok();
    }
}

impl Default for IncrementalChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Walk directory recursively and return all file paths
fn walkdir(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    fn walk_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    // Skip hidden dirs and target
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if !name.starts_with('.') && name != "target" {
                        walk_recursive(&path, files)?;
                    }
                } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    walk_recursive(root, &mut files)?;
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_hash() {
        // Create a temp file
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("bolt_test_hash.rs");
        std::fs::write(&test_file, "fn main() {}").unwrap();

        let hash1 = DependencyGraph::hash_file(&test_file);
        assert!(hash1.is_some());

        // Same content = same hash
        let hash2 = DependencyGraph::hash_file(&test_file);
        assert_eq!(hash1, hash2);

        // Different content = different hash
        std::fs::write(&test_file, "fn main() { println!(); }").unwrap();
        let hash3 = DependencyGraph::hash_file(&test_file);
        assert_ne!(hash1, hash3);

        std::fs::remove_file(&test_file).ok();
    }
}
