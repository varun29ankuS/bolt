//! External crate resolution
//!
//! Provides type information for external dependencies through two mechanisms:
//!
//! 1. **Stubs**: Pre-defined type signatures for common traits (fast, always available)
//! 2. **Rlib reading**: Extract metadata from compiled .rlib files (accurate, requires build)
//!
//! The resolver tries stubs first, then falls back to rlib reading.

pub mod rlib;
pub mod stubs;

use crate::cargo::Project;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

pub use stubs::{global_stubs, StubRegistry, TraitStub, TypeStub, MethodStub};
pub use rlib::{global_rlib_cache, RlibCache, RlibMetadata};

/// Result of resolving an external item
#[derive(Debug, Clone)]
pub enum ResolvedItem {
    /// Found in stubs
    Stub(StubItem),
    /// Found in compiled .rlib
    Rlib(RlibItem),
    /// Not found
    NotFound,
}

#[derive(Debug, Clone)]
pub struct StubItem {
    pub crate_name: String,
    pub item_name: String,
    pub kind: StubItemKind,
}

#[derive(Debug, Clone)]
pub enum StubItemKind {
    Trait(TraitStub),
    Type(TypeStub),
}

#[derive(Debug, Clone)]
pub struct RlibItem {
    pub crate_name: String,
    pub item_name: String,
    // Add more fields as rlib parsing improves
}

/// External crate resolver
///
/// Combines stubs and rlib reading to resolve external crate items.
#[derive(Default)]
pub struct ExternResolver {
    /// Project context (provides dependency info)
    project: Option<Arc<Project>>,
    /// Cache of resolved items
    resolved_cache: RwLock<HashMap<(String, String), ResolvedItem>>,
    /// Crates we've seen but couldn't resolve (avoid repeated lookups)
    unresolved_crates: RwLock<Vec<String>>,
}

impl ExternResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create resolver with project context
    pub fn with_project(project: Project) -> Self {
        Self {
            project: Some(Arc::new(project)),
            ..Default::default()
        }
    }

    /// Check if a name refers to an external crate
    pub fn is_external_crate(&self, name: &str) -> bool {
        // Check if it's a known external crate from project
        if let Some(ref project) = self.project {
            if project.crate_sources.contains_key(name) {
                return true;
            }
        }

        // Check common crate names that might have stubs
        matches!(name,
            "std" | "core" | "alloc" |
            "serde" | "serde_json" | "serde_yaml" |
            "tokio" | "async_std" | "futures" |
            "anyhow" | "thiserror" |
            "clap" | "structopt" |
            "log" | "tracing" |
            "rayon" |
            "indexmap" | "hashbrown" |
            "parking_lot" |
            "once_cell" | "lazy_static" |
            "regex" |
            "chrono" | "time" |
            "reqwest" | "hyper" |
            "syn" | "quote" | "proc_macro2"
        )
    }

    /// Resolve an item from an external crate
    ///
    /// `crate_name`: e.g., "serde"
    /// `item_path`: e.g., ["Serialize"] or ["de", "Deserialize"]
    pub fn resolve(&self, crate_name: &str, item_path: &[&str]) -> ResolvedItem {
        if item_path.is_empty() {
            return ResolvedItem::NotFound;
        }

        let item_name = item_path.last().unwrap();
        let cache_key = (crate_name.to_string(), item_name.to_string());

        // Check cache first
        if let Some(cached) = self.resolved_cache.read().get(&cache_key) {
            return cached.clone();
        }

        // Try stubs first (fast path)
        let stubs = global_stubs();

        // Check for trait
        if let Some(trait_stub) = stubs.get_trait(crate_name, item_name) {
            let result = ResolvedItem::Stub(StubItem {
                crate_name: crate_name.to_string(),
                item_name: item_name.to_string(),
                kind: StubItemKind::Trait(trait_stub.clone()),
            });
            self.resolved_cache.write().insert(cache_key, result.clone());
            return result;
        }

        // Check for type
        if let Some(type_stub) = stubs.get_type(crate_name, item_name) {
            let result = ResolvedItem::Stub(StubItem {
                crate_name: crate_name.to_string(),
                item_name: item_name.to_string(),
                kind: StubItemKind::Type(type_stub.clone()),
            });
            self.resolved_cache.write().insert(cache_key, result.clone());
            return result;
        }

        // Try rlib (slower path, but more accurate)
        let mut rlib_cache = global_rlib_cache().write();
        if let Some(_metadata) = rlib_cache.get(crate_name) {
            // TODO: Search metadata.exports for the item
            // For now, just mark that we found the crate
            let result = ResolvedItem::Rlib(RlibItem {
                crate_name: crate_name.to_string(),
                item_name: item_name.to_string(),
            });
            self.resolved_cache.write().insert(cache_key, result.clone());
            return result;
        }

        // Not found
        self.resolved_cache.write().insert(cache_key.clone(), ResolvedItem::NotFound);
        ResolvedItem::NotFound
    }

    /// Resolve a full path like "serde::Serialize" or "std::collections::HashMap"
    pub fn resolve_path(&self, path: &[&str]) -> ResolvedItem {
        if path.is_empty() {
            return ResolvedItem::NotFound;
        }

        let crate_name = path[0];
        if !self.is_external_crate(crate_name) {
            return ResolvedItem::NotFound;
        }

        self.resolve(crate_name, &path[1..])
    }

    /// Check if a trait is implemented for a type (stub-based)
    ///
    /// For stubs, we can't know actual implementations, so we return
    /// true for common auto-derived traits on simple types.
    pub fn check_trait_impl(&self, type_name: &str, trait_name: &str) -> bool {
        // For stub-based checking, assume common derives work
        // This is a heuristic - rustc does the real check
        match trait_name {
            "Clone" | "Copy" | "Debug" | "Default" |
            "PartialEq" | "Eq" | "PartialOrd" | "Ord" | "Hash" => {
                // These can be derived for most types
                true
            }
            "Serialize" | "Deserialize" => {
                // Assume serde works for types that use #[derive(Serialize)]
                true
            }
            "Send" | "Sync" => {
                // Assume most types are Send+Sync unless proven otherwise
                true
            }
            _ => false
        }
    }

    /// Get method signature for a trait method (from stub)
    pub fn get_trait_method(&self, crate_name: &str, trait_name: &str, method_name: &str)
        -> Option<&MethodStub>
    {
        let stubs = global_stubs();
        stubs.get_trait(crate_name, trait_name)
            .and_then(|t| t.methods.iter().find(|m| m.name == method_name))
    }

    /// List all available stub traits
    pub fn available_traits(&self) -> Vec<String> {
        global_stubs().all_trait_names().map(|s| s.to_string()).collect()
    }
}

/// Global extern resolver instance
pub fn global_resolver() -> &'static ExternResolver {
    use once_cell::sync::Lazy;
    static RESOLVER: Lazy<ExternResolver> = Lazy::new(ExternResolver::new);
    &RESOLVER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_serde() {
        let resolver = ExternResolver::new();

        // Should find Serialize from stubs
        let result = resolver.resolve("serde", &["Serialize"]);
        match result {
            ResolvedItem::Stub(item) => {
                assert_eq!(item.item_name, "Serialize");
                assert_eq!(item.crate_name, "serde");
            }
            _ => panic!("Expected stub for Serialize"),
        }
    }

    #[test]
    fn test_resolve_std() {
        let resolver = ExternResolver::new();

        // Should find Clone from stubs
        let result = resolver.resolve("std", &["Clone"]);
        match result {
            ResolvedItem::Stub(item) => {
                assert_eq!(item.item_name, "Clone");
            }
            _ => panic!("Expected stub for Clone"),
        }
    }

    #[test]
    fn test_is_external_crate() {
        let resolver = ExternResolver::new();

        assert!(resolver.is_external_crate("serde"));
        assert!(resolver.is_external_crate("std"));
        assert!(resolver.is_external_crate("tokio"));
        assert!(!resolver.is_external_crate("my_local_mod"));
    }
}
