//! Parse dependency source code from cargo registry
//!
//! Instead of parsing complex .rlib/.rmeta files, we can parse the actual
//! source code of dependencies from ~/.cargo/registry/src/

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;

/// Parsed dependency information
#[derive(Debug, Clone, Default)]
pub struct ParsedCrate {
    pub name: String,
    pub version: String,
    /// Public types exported (name -> kind)
    pub types: HashMap<String, String>,
    /// Public traits exported (name -> methods)
    pub traits: HashMap<String, Vec<String>>,
    /// Public functions exported (name -> signature)
    pub functions: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub name: String,
    pub kind: String,
}

#[derive(Debug, Clone)]
pub struct TraitInfo {
    pub name: String,
    pub methods: Vec<String>,
}

/// Cache of parsed dependency crates
pub struct DependencyCache {
    /// Parsed crates by name
    crates: DashMap<String, Arc<ParsedCrate>>,
    /// Cargo registry source path
    registry_path: Option<PathBuf>,
}

impl DependencyCache {
    pub fn new() -> Self {
        let registry_path = find_cargo_registry();
        Self {
            crates: DashMap::new(),
            registry_path,
        }
    }

    /// Get or parse a dependency crate
    pub fn get_crate(&self, name: &str, _version: Option<&str>) -> Option<Arc<ParsedCrate>> {
        // Check cache first
        if let Some(cached) = self.crates.get(name) {
            return Some(cached.clone());
        }

        // Try to find and parse the crate
        if let Some(parsed) = self.parse_crate(name) {
            let arc = Arc::new(parsed);
            self.crates.insert(name.to_string(), arc.clone());
            return Some(arc);
        }

        None
    }

    /// Find and parse a crate from cargo registry (simplified - just extract names)
    fn parse_crate(&self, name: &str) -> Option<ParsedCrate> {
        let registry_path = self.registry_path.as_ref()?;

        // Find the crate directory
        let crate_dir = find_crate_in_registry(registry_path, name)?;

        // Parse src/lib.rs
        let lib_rs = crate_dir.join("src/lib.rs");
        if !lib_rs.exists() {
            return None;
        }

        // Simple regex-based extraction of public items
        let content = std::fs::read_to_string(&lib_rs).ok()?;
        Some(extract_public_items_simple(name, &content))
    }

    /// Resolve a type from a dependency
    pub fn resolve_type(&self, crate_name: &str, type_name: &str) -> Option<TypeInfo> {
        let krate = self.get_crate(crate_name, None)?;
        krate.types.get(type_name).map(|kind| TypeInfo {
            name: type_name.to_string(),
            kind: kind.clone(),
        })
    }

    /// Resolve a trait from a dependency
    pub fn resolve_trait(&self, crate_name: &str, trait_name: &str) -> Option<TraitInfo> {
        let krate = self.get_crate(crate_name, None)?;
        krate.traits.get(trait_name).map(|methods| TraitInfo {
            name: trait_name.to_string(),
            methods: methods.clone(),
        })
    }

    /// Check if a type implements a trait (stub-based for now)
    pub fn implements_trait(&self, _crate_name: &str, _type_name: &str, _trait_name: &str) -> bool {
        // Real implementation would need to parse impl blocks
        false
    }
}

impl Default for DependencyCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Find cargo registry source directory
fn find_cargo_registry() -> Option<PathBuf> {
    // Check CARGO_HOME first
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        let registry = PathBuf::from(cargo_home).join("registry/src");
        if registry.exists() {
            return Some(registry);
        }
    }

    // Default locations
    let home = dirs::home_dir()?;

    // Unix: ~/.cargo/registry/src
    let unix_path = home.join(".cargo/registry/src");
    if unix_path.exists() {
        return Some(unix_path);
    }

    // Windows: %USERPROFILE%\.cargo\registry\src
    let win_path = home.join(".cargo").join("registry").join("src");
    if win_path.exists() {
        return Some(win_path);
    }

    None
}

/// Find a specific crate in the registry
fn find_crate_in_registry(registry: &Path, name: &str) -> Option<PathBuf> {
    // Registry structure: registry/src/index.crates.io-*/cratename-version/
    for entry in std::fs::read_dir(registry).ok()?.flatten() {
        let index_dir = entry.path();
        if !index_dir.is_dir() {
            continue;
        }

        // Look for crate directories
        for crate_entry in std::fs::read_dir(&index_dir).ok()?.flatten() {
            let crate_dir = crate_entry.path();
            let dir_name = crate_dir.file_name()?.to_str()?;

            // Match "cratename-version" pattern
            if dir_name.starts_with(&format!("{}-", name)) {
                return Some(crate_dir);
            }
        }
    }

    None
}

/// Simple extraction of public items using string matching
fn extract_public_items_simple(name: &str, content: &str) -> ParsedCrate {
    let mut types = HashMap::new();
    let mut traits = HashMap::new();
    let mut functions = HashMap::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Match "pub struct Name"
        if trimmed.starts_with("pub struct ") {
            if let Some(name) = extract_ident(trimmed, "pub struct ") {
                types.insert(name, "struct".to_string());
            }
        }
        // Match "pub enum Name"
        else if trimmed.starts_with("pub enum ") {
            if let Some(name) = extract_ident(trimmed, "pub enum ") {
                types.insert(name, "enum".to_string());
            }
        }
        // Match "pub trait Name"
        else if trimmed.starts_with("pub trait ") {
            if let Some(name) = extract_ident(trimmed, "pub trait ") {
                traits.insert(name, Vec::new());
            }
        }
        // Match "pub fn name"
        else if trimmed.starts_with("pub fn ") {
            if let Some(name) = extract_ident(trimmed, "pub fn ") {
                functions.insert(name, trimmed.to_string());
            }
        }
        // Match "pub type Name"
        else if trimmed.starts_with("pub type ") {
            if let Some(name) = extract_ident(trimmed, "pub type ") {
                types.insert(name, "type_alias".to_string());
            }
        }
    }

    ParsedCrate {
        name: name.to_string(),
        version: String::new(),
        types,
        traits,
        functions,
    }
}

/// Extract identifier after a prefix
fn extract_ident(line: &str, prefix: &str) -> Option<String> {
    let rest = line.strip_prefix(prefix)?;
    let end = rest.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(rest.len());
    let ident = &rest[..end];
    if ident.is_empty() {
        None
    } else {
        Some(ident.to_string())
    }
}

/// Global dependency cache
static DEPENDENCY_CACHE: once_cell::sync::Lazy<DependencyCache> =
    once_cell::sync::Lazy::new(DependencyCache::new);

pub fn global_dependency_cache() -> &'static DependencyCache {
    &DEPENDENCY_CACHE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ident() {
        assert_eq!(extract_ident("pub struct Foo {", "pub struct "), Some("Foo".to_string()));
        assert_eq!(extract_ident("pub enum Bar", "pub enum "), Some("Bar".to_string()));
        assert_eq!(extract_ident("pub fn baz()", "pub fn "), Some("baz".to_string()));
    }

    #[test]
    fn test_find_registry() {
        let registry = find_cargo_registry();
        println!("Registry path: {:?}", registry);
    }
}
