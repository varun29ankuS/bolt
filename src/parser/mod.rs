//! Rust source code parser using syn
//!
//! Converts Rust source to HIR (High-level IR)

mod lower;

use crate::error::{BoltError, DiagnosticEmitter, Result, Span};
use crate::hir::{self, Crate, DefId};
use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

pub struct SourceMap {
    files: DashMap<u32, SourceFile>,
    next_id: AtomicU32,
}

impl SourceMap {
    pub fn new() -> Self {
        Self {
            files: DashMap::new(),
            next_id: AtomicU32::new(1),
        }
    }

    pub fn add_file(&self, path: PathBuf, content: String) -> u32 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let lines = compute_line_starts(&content);
        self.files.insert(id, SourceFile { path, content, lines });
        id
    }

    pub fn get_file(&self, id: u32) -> Option<dashmap::mapref::one::Ref<u32, SourceFile>> {
        self.files.get(&id)
    }

    pub fn lookup_line_col(&self, span: Span) -> Option<(PathBuf, usize, usize)> {
        let file = self.files.get(&span.file_id)?;
        let (line, col) = file.offset_to_line_col(span.start as usize);
        Some((file.path.clone(), line, col))
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SourceFile {
    pub path: PathBuf,
    pub content: String,
    lines: Vec<usize>,
}

impl SourceFile {
    fn offset_to_line_col(&self, offset: usize) -> (usize, usize) {
        let line = self.lines.partition_point(|&start| start <= offset);
        let line_start = if line > 0 { self.lines[line - 1] } else { 0 };
        let col = offset - line_start + 1;
        (line, col)
    }
}

fn compute_line_starts(content: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (i, c) in content.char_indices() {
        if c == '\n' {
            starts.push(i + 1);
        }
    }
    starts
}

pub struct Parser {
    source_map: SourceMap,
    diagnostics: RwLock<DiagnosticEmitter>,
    next_def_id: AtomicU32,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            source_map: SourceMap::new(),
            diagnostics: RwLock::new(DiagnosticEmitter::new()),
            next_def_id: AtomicU32::new(1),
        }
    }

    pub fn source_map(&self) -> &SourceMap {
        &self.source_map
    }

    fn alloc_def_id(&self) -> DefId {
        self.next_def_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn parse_crate(&self, root: &Path) -> Result<Crate> {
        let main_path = if root.is_dir() {
            root.join("src/main.rs")
        } else {
            root.to_path_buf()
        };

        if !main_path.exists() {
            return Err(BoltError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Entry point not found: {}", main_path.display()),
            )));
        }

        let crate_name = root
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("main")
            .to_string();

        let mut krate = Crate::new(crate_name);
        self.parse_file(&main_path, &mut krate)?;

        if self.diagnostics.read().has_errors() {
            let diags = self.diagnostics.write().take_diagnostics();
            let first_error = diags.into_iter().find(|d| d.level == crate::error::DiagnosticLevel::Error);
            if let Some(diag) = first_error {
                if let Some(span) = diag.span {
                    if let Some((path, line, col)) = self.source_map.lookup_line_col(span) {
                        return Err(BoltError::Parse {
                            file: path,
                            line,
                            col,
                            message: diag.message,
                        });
                    }
                }
                return Err(BoltError::Parse {
                    file: main_path,
                    line: 0,
                    col: 0,
                    message: diag.message,
                });
            }
        }

        Ok(krate)
    }

    fn parse_file(&self, path: &Path, krate: &mut Crate) -> Result<()> {
        let content = std::fs::read_to_string(path)?;
        let file_id = self.source_map.add_file(path.to_path_buf(), content.clone());

        let syn_file = syn::parse_file(&content).map_err(|e| BoltError::Parse {
            file: path.to_path_buf(),
            line: 1,
            col: 1,
            message: e.to_string(),
        })?;

        let lowerer = lower::Lowerer::new(self, file_id, path.to_path_buf());
        lowerer.lower_file(&syn_file, krate);

        Ok(())
    }

    pub fn parse_files_parallel(&self, paths: &[PathBuf]) -> Result<Vec<Crate>> {
        paths
            .par_iter()
            .map(|path| self.parse_crate(path))
            .collect()
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_starts() {
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        let starts = compute_line_starts(content);
        // Line 1: "fn main() {\n" = 12 chars, starts at 0
        // Line 2: "    println!(\"hello\");\n" = 23 chars, starts at 12
        // Line 3: "}\n" = 2 chars, starts at 35
        // Line 4 (after final newline): starts at 37
        assert_eq!(starts, vec![0, 12, 35, 37]);
    }
}
