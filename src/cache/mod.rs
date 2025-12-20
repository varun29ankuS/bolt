//! Incremental compilation cache
//!
//! Content-addressed cache using blake3 hashing.
//! Stores compiled artifacts keyed by source hash.

use crate::error::Result;
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub source_hash: String,
    pub timestamp: u64,
    pub dependencies: Vec<String>,
    pub artifact_path: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheIndex {
    pub entries: HashMap<String, CacheEntry>,
    pub version: u32,
}

impl Default for CacheIndex {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            version: 1,
        }
    }
}

pub struct Cache {
    root: PathBuf,
    index: CacheIndex,
}

impl Cache {
    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root)?;
        let index_path = root.join("index.json");

        let index = if index_path.exists() {
            let content = fs::read_to_string(&index_path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            CacheIndex::default()
        };

        Ok(Self { root, index })
    }

    pub fn hash_file(path: &Path) -> Result<String> {
        let content = fs::read(path)?;
        let mut hasher = Hasher::new();
        hasher.update(&content);
        Ok(hasher.finalize().to_hex().to_string())
    }

    pub fn hash_content(content: &[u8]) -> String {
        let mut hasher = Hasher::new();
        hasher.update(content);
        hasher.finalize().to_hex().to_string()
    }

    pub fn get(&self, key: &str) -> Option<&CacheEntry> {
        self.index.entries.get(key)
    }

    pub fn is_valid(&self, key: &str, current_hash: &str) -> bool {
        if let Some(entry) = self.index.entries.get(key) {
            entry.source_hash == current_hash && entry.artifact_path.exists()
        } else {
            false
        }
    }

    pub fn insert(&mut self, key: String, source_hash: String, dependencies: Vec<String>) -> Result<PathBuf> {
        let artifact_path = self.root.join(format!("{}.bolt", &source_hash[..16]));

        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let entry = CacheEntry {
            source_hash,
            timestamp,
            dependencies,
            artifact_path: artifact_path.clone(),
        };

        self.index.entries.insert(key, entry);
        self.save_index()?;

        Ok(artifact_path)
    }

    pub fn invalidate(&mut self, key: &str) -> Result<()> {
        if let Some(entry) = self.index.entries.remove(key) {
            if entry.artifact_path.exists() {
                fs::remove_file(&entry.artifact_path)?;
            }
        }
        self.save_index()?;
        Ok(())
    }

    pub fn invalidate_stale(&mut self) -> Result<usize> {
        let stale: Vec<_> = self
            .index
            .entries
            .iter()
            .filter(|(_, entry)| !entry.artifact_path.exists())
            .map(|(k, _)| k.clone())
            .collect();

        let count = stale.len();
        for key in stale {
            self.index.entries.remove(&key);
        }

        if count > 0 {
            self.save_index()?;
        }

        Ok(count)
    }

    pub fn clear(&mut self) -> Result<()> {
        for entry in self.index.entries.values() {
            if entry.artifact_path.exists() {
                let _ = fs::remove_file(&entry.artifact_path);
            }
        }
        self.index.entries.clear();
        self.save_index()?;
        Ok(())
    }

    fn save_index(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.index)?;
        fs::write(self.root.join("index.json"), content)?;
        Ok(())
    }

    pub fn stats(&self) -> CacheStats {
        let total_entries = self.index.entries.len();
        let valid_entries = self
            .index
            .entries
            .values()
            .filter(|e| e.artifact_path.exists())
            .count();

        let total_size: u64 = self
            .index
            .entries
            .values()
            .filter_map(|e| fs::metadata(&e.artifact_path).ok())
            .map(|m| m.len())
            .sum();

        CacheStats {
            total_entries,
            valid_entries,
            total_size_bytes: total_size,
        }
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub total_entries: usize,
    pub valid_entries: usize,
    pub total_size_bytes: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.total_entries == 0 {
            0.0
        } else {
            self.valid_entries as f64 / self.total_entries as f64
        }
    }
}

pub struct DependencyTracker {
    deps: HashMap<String, Vec<String>>,
}

impl DependencyTracker {
    pub fn new() -> Self {
        Self {
            deps: HashMap::new(),
        }
    }

    pub fn add_dependency(&mut self, file: String, depends_on: String) {
        self.deps.entry(file).or_default().push(depends_on);
    }

    pub fn get_dependencies(&self, file: &str) -> Option<&[String]> {
        self.deps.get(file).map(|v| v.as_slice())
    }

    pub fn dependents_of(&self, file: &str) -> Vec<String> {
        self.deps
            .iter()
            .filter(|(_, deps)| deps.contains(&file.to_string()))
            .map(|(f, _)| f.clone())
            .collect()
    }

    pub fn invalidation_set(&self, changed: &str) -> Vec<String> {
        let mut result = vec![changed.to_string()];
        let mut queue = vec![changed.to_string()];

        while let Some(file) = queue.pop() {
            for dependent in self.dependents_of(&file) {
                if !result.contains(&dependent) {
                    result.push(dependent.clone());
                    queue.push(dependent);
                }
            }
        }

        result
    }
}

impl Default for DependencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_consistency() {
        let content = b"fn main() { println!(\"hello\"); }";
        let hash1 = Cache::hash_content(content);
        let hash2 = Cache::hash_content(content);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_differs() {
        let hash1 = Cache::hash_content(b"fn main() {}");
        let hash2 = Cache::hash_content(b"fn main() { 1 }");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_dependency_tracker() {
        let mut tracker = DependencyTracker::new();
        tracker.add_dependency("a.rs".to_string(), "b.rs".to_string());
        tracker.add_dependency("b.rs".to_string(), "c.rs".to_string());

        let deps = tracker.get_dependencies("a.rs").unwrap();
        assert_eq!(deps, &["b.rs"]);

        let invalidated = tracker.invalidation_set("c.rs");
        assert!(invalidated.contains(&"c.rs".to_string()));
        assert!(invalidated.contains(&"b.rs".to_string()));
    }
}
