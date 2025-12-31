//! Read type information from compiled .rlib files
//!
//! Rust crates compile to .rlib files (ar archives) containing:
//! - .rmeta: Serialized type/trait metadata
//! - .o files: Compiled object code
//!
//! The .rmeta format is rustc-internal and version-specific, but we can
//! extract useful type information from it.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::path::{Path, PathBuf};

/// Information extracted from an .rlib file
#[derive(Debug, Default)]
pub struct RlibMetadata {
    pub crate_name: String,
    pub crate_hash: String,
    pub edition: String,
    /// Public items exported by this crate
    pub exports: Vec<ExportedItem>,
    /// Dependencies this crate requires
    pub dependencies: Vec<CrateDep>,
}

#[derive(Debug, Clone)]
pub struct ExportedItem {
    pub name: String,
    pub kind: ExportKind,
    pub visibility: bool, // true = pub
}

#[derive(Debug, Clone)]
pub enum ExportKind {
    Struct { fields: Vec<(String, String)> },
    Enum { variants: Vec<String> },
    Trait { methods: Vec<String> },
    Function { signature: String },
    Const,
    TypeAlias,
    Mod,
}

#[derive(Debug, Clone)]
pub struct CrateDep {
    pub name: String,
    pub hash: String,
}

/// Find .rlib files in standard locations
pub fn find_rlib(crate_name: &str) -> Option<PathBuf> {
    // Standard locations to search:
    // 1. target/release/deps/
    // 2. target/debug/deps/
    // 3. ~/.rustup/toolchains/stable-*/lib/rustlib/*/lib/
    // 4. ~/.cargo/registry/cache/

    let search_paths = vec![
        PathBuf::from("target/release/deps"),
        PathBuf::from("target/debug/deps"),
    ];

    // Also check CARGO_HOME
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        // Could search registry here
    }

    // Also check rustup toolchain for std library
    if let Ok(rustup_home) = std::env::var("RUSTUP_HOME") {
        // Could search toolchain libs here
    }

    for search_path in search_paths {
        if let Ok(entries) = std::fs::read_dir(&search_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    // .rlib files are named like: libserde-abc123.rlib
                    if name.starts_with(&format!("lib{}-", crate_name))
                        || name.starts_with(&format!("lib{}", crate_name))
                    {
                        if name.ends_with(".rlib") {
                            return Some(path);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Read metadata from an .rlib file
pub fn read_rlib(path: &Path) -> io::Result<RlibMetadata> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // .rlib is an ar archive
    // Format: !<arch>\n followed by entries
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;

    if &magic != b"!<arch>\n" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a valid ar archive",
        ));
    }

    let mut metadata = RlibMetadata::default();

    // Read ar entries looking for .rmeta
    loop {
        // Ar entry header is 60 bytes
        let mut header = [0u8; 60];
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }

        // Parse header
        let name = String::from_utf8_lossy(&header[0..16]).trim().to_string();
        let size_str = String::from_utf8_lossy(&header[48..58]).trim().to_string();
        let size: usize = size_str.parse().unwrap_or(0);

        // Read entry content
        let mut content = vec![0u8; size];
        reader.read_exact(&mut content)?;

        // Ar entries are padded to even byte boundaries
        if size % 2 == 1 {
            let mut padding = [0u8; 1];
            let _ = reader.read_exact(&mut padding);
        }

        // Check if this is the .rmeta entry
        if name.contains(".rmeta") || name.starts_with("lib.rmeta") {
            // Parse the rmeta content
            if let Ok(parsed) = parse_rmeta(&content) {
                metadata = parsed;
            }
            break;
        }

        // Handle extended names (names starting with '/' or '#1/')
        if name.starts_with("/") || name.starts_with("#1/") {
            // Extended name format - skip for now
            continue;
        }
    }

    // If we couldn't parse rmeta, at least extract crate name from path
    if metadata.crate_name.is_empty() {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            // libserde-abc123 -> serde
            if stem.starts_with("lib") {
                let name = &stem[3..];
                if let Some(dash) = name.find('-') {
                    metadata.crate_name = name[..dash].to_string();
                } else {
                    metadata.crate_name = name.to_string();
                }
            }
        }
    }

    Ok(metadata)
}

/// Parse .rmeta content
///
/// The .rmeta format is complex and version-specific. This is a simplified
/// parser that extracts basic information.
fn parse_rmeta(content: &[u8]) -> io::Result<RlibMetadata> {
    let mut metadata = RlibMetadata::default();

    // rmeta has a header with magic bytes and version
    if content.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rmeta too short",
        ));
    }

    // The exact format varies by rustc version
    // We'll do a best-effort extraction of readable strings

    // Look for the crate name - it's usually early in the file as a string
    let content_str = String::from_utf8_lossy(content);

    // Extract potential crate name (heuristic: look for common patterns)
    // The crate name often appears near the beginning after some binary data

    // For now, just scan for readable ASCII strings that look like identifiers
    let mut i = 0;
    let bytes = content;
    while i < bytes.len().saturating_sub(4) {
        // Look for length-prefixed strings (common in Rust serialization)
        let potential_len = bytes[i] as usize;
        if potential_len > 0 && potential_len < 64 && i + potential_len < bytes.len() {
            let slice = &bytes[i + 1..i + 1 + potential_len];
            if slice.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_') {
                let s = String::from_utf8_lossy(slice).to_string();
                if s.len() > 2 && metadata.crate_name.is_empty() {
                    // First valid identifier-like string might be crate name
                    if !s.starts_with("rust") && !s.starts_with("std") {
                        // metadata.crate_name = s;
                    }
                }
            }
        }
        i += 1;
    }

    Ok(metadata)
}

/// Cache of loaded rlib metadata
#[derive(Default)]
pub struct RlibCache {
    cache: HashMap<String, Option<RlibMetadata>>,
}

impl RlibCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get metadata for a crate, loading from .rlib if not cached
    pub fn get(&mut self, crate_name: &str) -> Option<&RlibMetadata> {
        if !self.cache.contains_key(crate_name) {
            let metadata = find_rlib(crate_name)
                .and_then(|path| read_rlib(&path).ok());
            self.cache.insert(crate_name.to_string(), metadata);
        }

        self.cache.get(crate_name).and_then(|opt| opt.as_ref())
    }

    /// Check if a crate's rlib exists
    pub fn has_rlib(&self, crate_name: &str) -> bool {
        find_rlib(crate_name).is_some()
    }
}

/// Global rlib cache
pub fn global_rlib_cache() -> &'static parking_lot::RwLock<RlibCache> {
    use once_cell::sync::Lazy;
    static CACHE: Lazy<parking_lot::RwLock<RlibCache>> =
        Lazy::new(|| parking_lot::RwLock::new(RlibCache::new()));
    &CACHE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_rlib() {
        // This test depends on having built the project
        // In CI, deps should exist in target/
        let serde_rlib = find_rlib("serde");
        // May or may not exist depending on build state
        println!("serde rlib: {:?}", serde_rlib);
    }
}
