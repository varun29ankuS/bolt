//! Cargo integration using cargo_metadata
//!
//! Provides full cargo project support including dependency resolution.

use cargo_metadata::{MetadataCommand, Package, Metadata};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Resolved project with full dependency information
#[derive(Debug)]
pub struct Project {
    pub name: String,
    pub version: String,
    pub root: PathBuf,
    pub entry_point: PathBuf,
    pub kind: ProjectKind,
    pub edition: String,
    /// All dependencies with their source paths
    pub dependencies: Vec<ResolvedDependency>,
    /// Map of crate name -> source files for type resolution
    pub crate_sources: HashMap<String, PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProjectKind {
    Library,
    Binary,
}

/// A resolved dependency with source location
#[derive(Debug, Clone)]
pub struct ResolvedDependency {
    pub name: String,
    pub version: String,
    /// Path to the crate's lib.rs or main source file
    pub source_path: Option<PathBuf>,
    /// Is this a path dependency (local)?
    pub is_local: bool,
}

/// Find Cargo.toml starting from a path and walking up
pub fn find_cargo_toml(start: &Path) -> Option<PathBuf> {
    let mut current = if start.is_file() {
        start.parent()?.to_path_buf()
    } else {
        start.to_path_buf()
    };

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            return Some(cargo_toml);
        }

        match current.parent() {
            Some(parent) => current = parent.to_path_buf(),
            None => return None,
        }
    }
}

/// Resolve a project using cargo_metadata for full dependency info
pub fn resolve_project(path: &Path) -> Result<Project, String> {
    // If path is a .rs file, just use it directly (single file mode)
    if path.extension().map(|e| e == "rs").unwrap_or(false) {
        return Ok(Project {
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            version: "0.0.0".to_string(),
            root: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            entry_point: path.to_path_buf(),
            kind: if path.file_name().map(|n| n == "lib.rs").unwrap_or(false) {
                ProjectKind::Library
            } else {
                ProjectKind::Binary
            },
            dependencies: vec![],
            crate_sources: HashMap::new(),
            edition: "2021".to_string(),
        });
    }

    // Find Cargo.toml
    let manifest_path = find_cargo_toml(path)
        .ok_or_else(|| format!("No Cargo.toml found in {} or parent directories", path.display()))?;

    // Run cargo metadata for full dependency resolution
    let metadata = MetadataCommand::new()
        .manifest_path(&manifest_path)
        .exec()
        .map_err(|e| format!("Failed to run cargo metadata: {}", e))?;

    // Find the root package
    let root_package = metadata.root_package()
        .ok_or("No root package found in workspace")?;

    // Determine entry point
    let (entry_point, kind) = find_entry_point(root_package)?;

    // Resolve all dependencies with their source paths
    let (dependencies, crate_sources) = resolve_dependencies(&metadata, root_package);

    Ok(Project {
        name: root_package.name.clone(),
        version: root_package.version.to_string(),
        root: manifest_path.parent().unwrap().to_path_buf(),
        entry_point,
        kind,
        edition: root_package.edition.to_string(),
        dependencies,
        crate_sources,
    })
}

/// Find the entry point for a package
fn find_entry_point(package: &Package) -> Result<(PathBuf, ProjectKind), String> {
    // Check for lib target first
    for target in &package.targets {
        if target.kind.iter().any(|k| k == "lib" || k == "rlib" || k == "dylib") {
            return Ok((target.src_path.clone().into(), ProjectKind::Library));
        }
    }

    // Then check for bin target
    for target in &package.targets {
        if target.kind.iter().any(|k| k == "bin") {
            return Ok((target.src_path.clone().into(), ProjectKind::Binary));
        }
    }

    // Fallback to default paths
    let manifest_dir = package.manifest_path.parent()
        .ok_or("Invalid manifest path")?;

    let lib_rs = manifest_dir.join("src/lib.rs");
    if lib_rs.exists() {
        return Ok((lib_rs.into(), ProjectKind::Library));
    }

    let main_rs = manifest_dir.join("src/main.rs");
    if main_rs.exists() {
        return Ok((main_rs.into(), ProjectKind::Binary));
    }

    Err("No entry point found (src/lib.rs or src/main.rs)".to_string())
}

/// Resolve all dependencies and their source paths
fn resolve_dependencies(metadata: &Metadata, root_package: &Package) -> (Vec<ResolvedDependency>, HashMap<String, PathBuf>) {
    let mut dependencies = Vec::new();
    let mut crate_sources = HashMap::new();

    // Build a map of package id -> package for quick lookup
    let packages: HashMap<_, _> = metadata.packages.iter()
        .map(|p| (p.id.clone(), p))
        .collect();

    // Get direct dependencies from root package
    for dep in &root_package.dependencies {
        // Find the resolved package
        if let Some(pkg) = metadata.packages.iter().find(|p| p.name == dep.name) {
            let source_path = find_package_lib_path(pkg);
            let is_local = dep.path.is_some();

            dependencies.push(ResolvedDependency {
                name: dep.name.clone(),
                version: pkg.version.to_string(),
                source_path: source_path.clone(),
                is_local,
            });

            if let Some(path) = source_path {
                crate_sources.insert(dep.name.clone(), path);
            }
        }
    }

    (dependencies, crate_sources)
}

/// Find the lib.rs path for a package
fn find_package_lib_path(package: &Package) -> Option<PathBuf> {
    // First try to find a lib target
    for target in &package.targets {
        if target.kind.iter().any(|k| k == "lib" || k == "rlib" || k == "proc-macro") {
            return Some(target.src_path.clone().into());
        }
    }

    // Fallback: check common paths
    let manifest_dir = package.manifest_path.parent()?;
    let lib_rs = manifest_dir.join("src/lib.rs");
    if lib_rs.exists() {
        return Some(lib_rs.into());
    }

    None
}

/// Quick check if a crate name is an external dependency
pub fn is_external_crate(name: &str, project: &Project) -> bool {
    project.crate_sources.contains_key(name)
}

/// Get the source path for an external crate
pub fn get_crate_source<'a>(name: &str, project: &'a Project) -> Option<&'a PathBuf> {
    project.crate_sources.get(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_cargo_toml() {
        let result = find_cargo_toml(Path::new("."));
        assert!(result.is_some());
    }
}
