//! Command-line interface for Bolt

use clap::{Parser as ClapParser, Subcommand, ValueEnum};
use colored::Colorize;
use std::path::PathBuf;
use std::time::Instant;

use crate::error::{
    DiagnosticLevel, DiagnosticReport, ErrorCode, JsonDiagnostic, SourceLocation,
};
use crate::ty::TypeRegistry;
use std::sync::Arc;

#[derive(ClapParser)]
#[command(name = "bolt")]
#[command(about = "⚡ Lightning-fast Rust type checker - 10-100x faster than cargo check")]
#[command(version)]
#[command(long_about = "Bolt is a fast type checker for Rust, optimized for the edit-check loop.

Get instant feedback on your code in milliseconds instead of seconds.
Use rustc/cargo for production builds - use Bolt for development speed.")]
#[command(after_help = "EXAMPLES:
    bolt check .              Type-check entire project (auto-detects Cargo.toml)
    bolt check src/lib.rs     Type-check a single file
    bolt check . -f json      JSON output for IDE/tooling integration
    bolt run main.rs          Compile and run instantly
    bolt watch src/ --run     Auto-recompile on file changes

TYPICAL WORKFLOW:
    1. Write code
    2. bolt check .           (~400ms feedback)
    3. Fix errors
    4. bolt check .           (~400ms feedback)
    5. cargo build --release  (final production build)

More info: https://github.com/varun29ankuS/bolt")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Output format for diagnostics
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable colored output (default)
    #[default]
    Human,
    /// JSON output optimized for LLM/AI tool consumption
    Json,
    /// Pretty-printed JSON for debugging
    JsonPretty,
}

/// Parser backend to use
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum ParserBackend {
    /// Syn-based parser (legacy)
    Syn,
    /// Chumsky-based parser with error recovery (default)
    #[default]
    Chumsky,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Build and run a Rust file or project
    Run {
        /// Path to Rust file or project
        path: PathBuf,

        /// Arguments to pass to the program
        #[arg(last = true)]
        args: Vec<String>,

        /// Output format for errors/diagnostics
        #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,

        /// Run borrow checker synchronously (default is async - code runs immediately)
        #[arg(long)]
        sync_borrow: bool,

        /// Parser backend to use
        #[arg(long, short = 'p', value_enum, default_value_t = ParserBackend::Syn)]
        parser: ParserBackend,
    },

    /// Build a Rust file or project
    Build {
        /// Path to Rust file or project
        path: PathBuf,

        /// Output path for the binary
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Release mode (optimized)
        #[arg(long)]
        release: bool,

        /// Output format for errors/diagnostics
        #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,

        /// Parser backend to use
        #[arg(long, short = 'p', value_enum, default_value_t = ParserBackend::Syn)]
        parser: ParserBackend,
    },

    /// Check a Rust file for errors (no codegen)
    Check {
        /// Path to Rust file or project
        path: PathBuf,

        /// Output format for errors/diagnostics
        #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,

        /// Parser backend to use
        #[arg(long, short = 'p', value_enum, default_value_t = ParserBackend::Syn)]
        parser: ParserBackend,
    },

    /// Show cache statistics
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// Watch files and recompile on changes (concurrent compilation)
    Watch {
        /// Path to Rust file or directory to watch
        path: PathBuf,

        /// Output format for errors/diagnostics
        #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,

        /// Run the program after each successful compilation
        #[arg(long, short)]
        run: bool,

        /// Parser backend to use
        #[arg(long, short = 'p', value_enum, default_value_t = ParserBackend::Syn)]
        parser: ParserBackend,
    },
}

#[derive(Subcommand)]
pub enum CacheAction {
    /// Show cache statistics
    Stats,
    /// Clear the cache
    Clear,
    /// Remove stale entries
    Clean,
}

pub fn run_cli() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run { path, args, format, sync_borrow, parser } => {
            run_file(&path, &args, format, !sync_borrow, parser);
        }
        Commands::Build { path, output, release, format, parser } => {
            build_file(&path, output.as_deref(), release, format, parser);
        }
        Commands::Check { path, format, parser } => {
            check_file(&path, format, parser);
        }
        Commands::Cache { action } => {
            handle_cache(action);
        }
        Commands::Watch { path, format, run, parser } => {
            watch_file(&path, format, run, parser);
        }
    }
}

// ============================================================================
// Diagnostic Output Helpers
// ============================================================================

/// Emit a single error and exit
fn emit_error_and_exit(
    format: OutputFormat,
    code: ErrorCode,
    message: &str,
    file: Option<&PathBuf>,
    line: Option<usize>,
    col: Option<usize>,
) -> ! {
    let diag = JsonDiagnostic::error(code, message);
    let diag = if let (Some(f), Some(l), Some(c)) = (file, line, col) {
        diag.with_location(SourceLocation {
            file: f.clone(),
            line: l,
            column: c,
            end_line: None,
            end_column: None,
        })
    } else {
        diag
    };

    emit_diagnostics(format, vec![diag]);
    print_rustc_fallback(format, file);
    std::process::exit(1);
}

/// Emit diagnostics in the appropriate format
fn emit_diagnostics(format: OutputFormat, diagnostics: Vec<JsonDiagnostic>) {
    match format {
        OutputFormat::Human => {
            for diag in &diagnostics {
                let label = match diag.level {
                    DiagnosticLevel::Error => "Error".red().bold(),
                    DiagnosticLevel::Warning => "Warning".yellow().bold(),
                    DiagnosticLevel::Note => "Note".blue().bold(),
                };

                if let Some(loc) = &diag.location {
                    eprintln!(
                        "{}: {}:{}:{}: {}",
                        label,
                        loc.file.display(),
                        loc.line,
                        loc.column,
                        diag.message
                    );
                } else {
                    eprintln!("{}: {}", label, diag.message);
                }

                // Print source snippet if available
                if let Some(snippet) = &diag.source_snippet {
                    eprintln!("  {}", snippet);
                }

                // Print notes
                for note in &diag.notes {
                    eprintln!("  {} {}", "=".cyan(), note);
                }

                // Print suggestions
                for suggestion in &diag.suggestions {
                    eprintln!(
                        "  {} {}: {}",
                        "help:".green().bold(),
                        suggestion.message,
                        suggestion.replacement
                    );
                }

                // Print help
                if let Some(help) = &diag.help {
                    eprintln!("  {} {}", "help:".green().bold(), help);
                }
            }
        }
        OutputFormat::Json => {
            let report = DiagnosticReport::new(diagnostics);
            println!("{}", report.to_json());
        }
        OutputFormat::JsonPretty => {
            let report = DiagnosticReport::new(diagnostics);
            println!("{}", report.to_json_pretty());
        }
    }
}

/// Print rustc fallback suggestion
fn print_rustc_fallback(format: OutputFormat, file: Option<&PathBuf>) {
    if matches!(format, OutputFormat::Human) {
        eprintln!();
        if let Some(path) = file {
            eprintln!(
                "  {} Bolt couldn't handle this. Try: {} {}",
                "💡".yellow(),
                "rustc --edition 2021".cyan(),
                path.display()
            );
        } else {
            eprintln!(
                "  {} Bolt couldn't handle this. Try: {}",
                "💡".yellow(),
                "cargo check".cyan()
            );
        }
    }
}

/// Emit diagnostics with rustc fallback suggestion, then exit
fn emit_diagnostics_and_exit(format: OutputFormat, diagnostics: Vec<JsonDiagnostic>, file: Option<&PathBuf>) -> ! {
    emit_diagnostics(format, diagnostics);
    print_rustc_fallback(format, file);
    std::process::exit(1);
}

/// Parse source code using the selected backend
fn parse_with_backend(
    path: &PathBuf,
    parser: ParserBackend,
    format: OutputFormat,
) -> Result<crate::hir::Crate, String> {
    match parser {
        ParserBackend::Syn => {
            let syn_parser = crate::parser::Parser::new();
            syn_parser.parse_crate(path).map_err(|e| e.to_string())
        }
        ParserBackend::Chumsky => {
            // Read source file
            let source = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read file: {}", e))?;

            // Get crate name from file stem
            let crate_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            if matches!(format, OutputFormat::Human) {
                eprintln!("  {} Using Chumsky parser (experimental)", "⚡".yellow());
            }

            crate::parser2::lower::parse_and_lower(&source, crate_name)
                .map_err(|errors| errors.join("\n"))
        }
    }
}

/// Emit success message in appropriate format
fn emit_success(format: OutputFormat, message: &str, details: Option<serde_json::Value>) {
    match format {
        OutputFormat::Human => {
            println!("{}", message.green().bold());
        }
        OutputFormat::Json | OutputFormat::JsonPretty => {
            let output = serde_json::json!({
                "status": "success",
                "message": message,
                "details": details
            });
            if matches!(format, OutputFormat::JsonPretty) {
                println!("{}", serde_json::to_string_pretty(&output).unwrap());
            } else {
                println!("{}", serde_json::to_string(&output).unwrap());
            }
        }
    }
}

fn run_file(path: &PathBuf, _args: &[String], format: OutputFormat, async_mode: bool, parser_backend: ParserBackend) {
    let start = Instant::now();

    // Check if previous async borrow check failed - if so, block and show errors
    if async_mode {
        let checker = crate::borrowck::global_checker();
        if let Some(diagnostics) = checker.check_previous_failure(path) {
            if matches!(format, OutputFormat::Human) {
                eprintln!(
                    "{} Previous borrow check found issues:",
                    "Blocked:".red().bold()
                );
            }
            let json_diags: Vec<JsonDiagnostic> = diagnostics
                .into_iter()
                .map(|d| {
                    JsonDiagnostic::error(ErrorCode::BorrowOfMovedValue, &d.message)
                        .with_notes_from_vec(d.notes)
                })
                .collect();
            emit_diagnostics_and_exit(format, json_diags, Some(path));
        }
    }

    if matches!(format, OutputFormat::Human) {
        println!("{} {} {}", "⚡".yellow(), "Compiling".green().bold(), path.display());
    }

    let krate = match parse_with_backend(path, parser_backend, format) {
        Ok(k) => k,
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::UnexpectedToken, &e, Some(path), None, None);
        }
    };

    // Create type registry and initialize with crate definitions
    let registry = Arc::new(TypeRegistry::new());
    registry.init_from_crate(&krate);

    let type_ctx = crate::typeck::TypeContext::new(Arc::clone(&registry));
    let mut type_checker = crate::typeck::TypeChecker::new(&type_ctx, &krate);
    if let Err(e) = type_checker.check_crate() {
        emit_error_and_exit(format, ErrorCode::TypeMismatch, &e.to_string(), Some(path), None, None);
    }

    if type_ctx.has_errors() {
        let diagnostics: Vec<JsonDiagnostic> = type_ctx
            .take_diagnostics()
            .into_iter()
            .map(|d| JsonDiagnostic {
                level: d.level,
                code: ErrorCode::TypeMismatch,
                message: d.message,
                location: None,
                source_snippet: None,
                labels: vec![],
                notes: d.notes,
                suggestions: vec![],
                help: None,
            })
            .collect();
        emit_diagnostics(format, diagnostics);
        emit_error_and_exit(format, ErrorCode::TypeMismatch, "Type checking failed", Some(path), None, None);
    }

    // Borrow checking: async or sync based on flag
    let krate_arc = Arc::new(krate);

    // Extract type aliases for borrow checker to resolve Copy types correctly
    // e.g., DefId = u32 should be recognized as Copy
    let type_alias_map = registry.get_type_alias_map();

    if async_mode {
        // Spawn borrow check in background - don't wait for it
        let checker = crate::borrowck::global_checker();
        checker.spawn_check_with_aliases(path.clone(), Arc::clone(&krate_arc), type_alias_map);
    } else {
        // Synchronous borrow checking (traditional mode)
        let borrow_checker = crate::borrowck::BorrowChecker::with_type_aliases(type_alias_map.clone());
        borrow_checker.check_crate(&krate_arc);
        if borrow_checker.has_errors() {
            let diagnostics: Vec<JsonDiagnostic> = borrow_checker
                .take_diagnostics()
                .into_iter()
                .map(|d| {
                    JsonDiagnostic::error(ErrorCode::BorrowOfMovedValue, &d.message)
                        .with_notes_from_vec(d.notes)
                })
                .collect();
            emit_diagnostics_and_exit(format, diagnostics, Some(path));
        }
    }

    let mut codegen = match crate::codegen::CodeGenerator::new(Arc::clone(&registry)) {
        Ok(c) => c,
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::InternalError, &e.to_string(), None, None, None);
        }
    };

    if let Err(e) = codegen.compile_crate(&krate_arc) {
        emit_error_and_exit(format, ErrorCode::InternalError, &e.to_string(), Some(path), None, None);
    }

    let compile_time = start.elapsed();

    if matches!(format, OutputFormat::Human) {
        println!(
            "{} {} in {:.2}ms{}",
            "⚡".yellow(),
            "Compiled".green().bold(),
            compile_time.as_secs_f64() * 1000.0,
            if async_mode { "" } else { " (sync)" }
        );
        println!("{} {}", "⚡".yellow(), "Running...".cyan());
        println!();
    }

    match codegen.run_main() {
        Ok(exit_code) => {
            if matches!(format, OutputFormat::Human) {
                println!();
                println!(
                    "{} {} with code {}",
                    "⚡".yellow(),
                    "Finished".green().bold(),
                    exit_code
                );
            }

            // After execution, check if async borrow check completed and show results
            if async_mode {
                let checker = crate::borrowck::global_checker();
                match checker.wait_for_result(path) {
                    crate::borrowck::CheckResult::Failed(diagnostics) => {
                        if matches!(format, OutputFormat::Human) {
                            eprintln!();
                            eprintln!(
                                "{} Borrow checker found issues (code ran anyway):",
                                "Warning:".yellow().bold()
                            );
                        }
                        let json_diags: Vec<JsonDiagnostic> = diagnostics
                            .into_iter()
                            .map(|d| {
                                JsonDiagnostic::warning(ErrorCode::BorrowOfMovedValue, &d.message)
                                    .with_notes_from_vec(d.notes)
                            })
                            .collect();
                        emit_diagnostics(format, json_diags);
                        // Note: Don't exit with error - code already ran
                        // Next compilation will block if this file is run again
                    }
                    crate::borrowck::CheckResult::Success => {
                        // All good
                    }
                    crate::borrowck::CheckResult::Pending => {
                        // Shouldn't happen after wait_for_result
                    }
                }
            }

            if !async_mode || matches!(format, OutputFormat::Json | OutputFormat::JsonPretty) {
                emit_success(format, "Execution completed", Some(serde_json::json!({
                    "exit_code": exit_code,
                    "compile_time_ms": compile_time.as_secs_f64() * 1000.0,
                    "async_mode": async_mode
                })));
            }
        }
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::InternalError, &format!("Runtime error: {}", e), None, None, None);
        }
    }
}

fn build_file(path: &PathBuf, _output: Option<&std::path::Path>, release: bool, format: OutputFormat, parser_backend: ParserBackend) {
    let start = Instant::now();

    let mode = if release { "release" } else { "debug" };
    if matches!(format, OutputFormat::Human) {
        println!(
            "{} {} {} [{}]",
            "⚡".yellow(),
            "Compiling".green().bold(),
            path.display(),
            mode
        );
    }

    let krate = match parse_with_backend(path, parser_backend, format) {
        Ok(k) => k,
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::UnexpectedToken, &e, Some(path), None, None);
        }
    };

    // Create type registry and initialize with crate definitions
    let registry = Arc::new(TypeRegistry::new());
    registry.init_from_crate(&krate);

    let type_ctx = crate::typeck::TypeContext::new(Arc::clone(&registry));
    let mut type_checker = crate::typeck::TypeChecker::new(&type_ctx, &krate);
    if let Err(e) = type_checker.check_crate() {
        emit_error_and_exit(format, ErrorCode::TypeMismatch, &e.to_string(), Some(path), None, None);
    }

    // Extract type aliases for borrow checker to resolve Copy types correctly
    let type_alias_map = registry.get_type_alias_map();

    // Run both fast heuristic checker and full NLL analysis for maximum accuracy
    let borrow_checker = crate::borrowck::BorrowChecker::with_type_aliases(type_alias_map.clone());
    borrow_checker.check_crate(&krate);
    let fast_diags = borrow_checker.take_diagnostics();

    // Full NLL analysis (CFG-based, catches more edge cases)
    let nll_checker = crate::borrowck::NllChecker::new();
    nll_checker.check_crate(&krate);
    let nll_diags = nll_checker.take_diagnostics();

    // Merge and deduplicate diagnostics
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut all_diags = Vec::new();
    for d in fast_diags.into_iter().chain(nll_diags.into_iter()) {
        if seen.insert(d.message.clone()) {
            all_diags.push(d);
        }
    }

    if !all_diags.is_empty() {
        let diagnostics: Vec<JsonDiagnostic> = all_diags
            .into_iter()
            .map(|d| {
                JsonDiagnostic::error(ErrorCode::BorrowOfMovedValue, &d.message)
                    .with_notes_from_vec(d.notes)
            })
            .collect();
        emit_diagnostics_and_exit(format, diagnostics, Some(path));
    }

    let mut codegen = match crate::codegen::CodeGenerator::new(Arc::clone(&registry)) {
        Ok(c) => c,
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::InternalError, &e.to_string(), Some(path), None, None);
        }
    };

    if let Err(e) = codegen.compile_crate(&krate) {
        emit_error_and_exit(format, ErrorCode::InternalError, &e.to_string(), Some(path), None, None);
    }

    let compile_time = start.elapsed();

    if matches!(format, OutputFormat::Human) {
        println!(
            "{} {} in {:.2}ms",
            "⚡".yellow(),
            "Finished".green().bold(),
            compile_time.as_secs_f64() * 1000.0
        );
    } else {
        emit_success(format, "Build completed", Some(serde_json::json!({
            "mode": mode,
            "compile_time_ms": compile_time.as_secs_f64() * 1000.0
        })));
    }
}

fn check_file(path: &PathBuf, format: OutputFormat, parser_backend: ParserBackend) {
    let start = Instant::now();

    // Resolve project - handles directories, Cargo.toml, or direct .rs files
    let (entry_point, project_name) = if path.is_dir() || !path.extension().map(|e| e == "rs").unwrap_or(false) {
        match crate::cargo::resolve_project(path) {
            Ok(project) => {
                if matches!(format, OutputFormat::Human) {
                    println!("{} {} {} ({})",
                        "⚡".yellow(),
                        "Checking".cyan().bold(),
                        project.name.cyan(),
                        project.entry_point.display()
                    );
                    if !project.dependencies.is_empty() {
                        println!("  {} {} dependencies (stub resolution)",
                            "⚡".yellow(),
                            project.dependencies.len()
                        );
                    }
                }
                (project.entry_point, project.name)
            }
            Err(e) => {
                emit_error_and_exit(format, ErrorCode::IoError, &e, Some(path), None, None);
            }
        }
    } else {
        if matches!(format, OutputFormat::Human) {
            println!("{} {} {}", "⚡".yellow(), "Checking".cyan().bold(), path.display());
        }
        (path.clone(), path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string())
    };

    let krate = match parse_with_backend(&entry_point, parser_backend, format) {
        Ok(k) => k,
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::UnexpectedToken, &e, Some(&entry_point), None, None);
        }
    };

    let parse_time = start.elapsed();

    let type_start = Instant::now();
    // Create type registry and initialize with crate definitions
    let registry = Arc::new(TypeRegistry::new());
    registry.init_from_crate(&krate);

    let type_ctx = crate::typeck::TypeContext::new(Arc::clone(&registry));
    let mut type_checker = crate::typeck::TypeChecker::new(&type_ctx, &krate);
    if let Err(e) = type_checker.check_crate() {
        emit_error_and_exit(format, ErrorCode::TypeMismatch, &e.to_string(), Some(&entry_point), None, None);
    }
    let type_time = type_start.elapsed();

    let borrow_start = Instant::now();
    // Extract type aliases for borrow checker to resolve Copy types correctly
    let type_alias_map = registry.get_type_alias_map();

    // Run both fast heuristic checker and full NLL analysis for maximum accuracy
    let borrow_checker = crate::borrowck::BorrowChecker::with_type_aliases(type_alias_map.clone());
    borrow_checker.check_crate(&krate);
    let fast_diags = borrow_checker.take_diagnostics();

    // Full NLL analysis (CFG-based, catches more edge cases)
    let nll_checker = crate::borrowck::NllChecker::new();
    nll_checker.check_crate(&krate);
    let nll_diags = nll_checker.take_diagnostics();

    // Merge and deduplicate diagnostics
    let mut seen_msgs: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut all_borrow_diags = Vec::new();
    for d in fast_diags.into_iter().chain(nll_diags.into_iter()) {
        if seen_msgs.insert(d.message.clone()) {
            all_borrow_diags.push(d);
        }
    }
    let borrow_time = borrow_start.elapsed();

    let has_errors = type_ctx.has_errors() || !all_borrow_diags.is_empty();

    if has_errors {
        // Collect type checker diagnostics
        let mut diagnostics: Vec<JsonDiagnostic> = type_ctx
            .take_diagnostics()
            .into_iter()
            .map(|d| {
                JsonDiagnostic::error(ErrorCode::TypeMismatch, &d.message)
                    .with_notes_from_vec(d.notes)
            })
            .collect();
        // Collect borrow checker diagnostics (already merged fast + NLL)
        diagnostics.extend(all_borrow_diags
            .into_iter()
            .map(|d| {
                JsonDiagnostic::error(ErrorCode::BorrowOfMovedValue, &d.message)
                    .with_notes_from_vec(d.notes)
            }));
        emit_diagnostics_and_exit(format, diagnostics, Some(&entry_point));
    }

    let total_time = start.elapsed();

    if matches!(format, OutputFormat::Human) {
        println!();
        println!("{} {}", "⚡".yellow(), "No errors found!".green().bold());
        println!();
        println!("  {} Parse:  {:>6.2}ms", "⚡".yellow(), parse_time.as_secs_f64() * 1000.0);
        println!("  {} Type:   {:>6.2}ms", "⚡".yellow(), type_time.as_secs_f64() * 1000.0);
        println!("  {} Borrow: {:>6.2}ms", "⚡".yellow(), borrow_time.as_secs_f64() * 1000.0);
        println!("  ──────────────────────");
        println!(
            "  {} Total:  {:>6.2}ms",
            "⚡".yellow(),
            total_time.as_secs_f64() * 1000.0
        );
    } else {
        emit_success(format, "Check completed - no errors", Some(serde_json::json!({
            "parse_time_ms": parse_time.as_secs_f64() * 1000.0,
            "type_time_ms": type_time.as_secs_f64() * 1000.0,
            "borrow_time_ms": borrow_time.as_secs_f64() * 1000.0,
            "total_time_ms": total_time.as_secs_f64() * 1000.0
        })));
    }
}

fn handle_cache(action: CacheAction) {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("bolt");

    let mut cache = match crate::cache::Cache::new(cache_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}: Failed to open cache: {}", "Error".red().bold(), e);
            std::process::exit(1);
        }
    };

    match action {
        CacheAction::Stats => {
            let stats = cache.stats();
            println!("{} {}", "⚡".yellow(), "Cache Statistics".cyan().bold());
            println!();
            println!("  {} Entries:    {}", "⚡".yellow(), stats.total_entries);
            println!("  {} Valid:      {}", "⚡".yellow(), stats.valid_entries);
            println!("  {} Hit rate:   {:.1}%", "⚡".yellow(), stats.hit_rate() * 100.0);
            println!("  {} Total size: {} KB", "⚡".yellow(), stats.total_size_bytes / 1024);
        }
        CacheAction::Clear => {
            match cache.clear() {
                Ok(()) => println!("{} {}", "⚡".yellow(), "Cache cleared".green().bold()),
                Err(e) => {
                    eprintln!("{}: {}", "Error".red().bold(), e);
                    std::process::exit(1);
                }
            }
        }
        CacheAction::Clean => {
            match cache.invalidate_stale() {
                Ok(count) => {
                    println!(
                        "{} {} {} stale entries",
                        "⚡".yellow(),
                        "Removed".green().bold(),
                        count
                    );
                }
                Err(e) => {
                    eprintln!("{}: {}", "Error".red().bold(), e);
                    std::process::exit(1);
                }
            }
        }
    }
}

// ============================================================================
// Watch Mode - Async Concurrent Compilation
// ============================================================================

fn watch_file(path: &PathBuf, format: OutputFormat, run_after: bool, parser_backend: ParserBackend) {
    use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
    use std::sync::mpsc::channel;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Duration;

    let path = path.canonicalize().unwrap_or_else(|_| path.clone());
    let watch_path = if path.is_file() {
        path.parent().unwrap_or(&path).to_path_buf()
    } else {
        path.clone()
    };

    if matches!(format, OutputFormat::Human) {
        println!(
            "{} {} {} for changes...",
            "⚡".yellow(),
            "Watching".cyan().bold(),
            watch_path.display()
        );
        println!("  {} Press Ctrl+C to stop", "⚡".yellow());
        println!("  {} Compilation runs in background - system stays responsive\n", "⚡".yellow());
    }

    // Shared state for async compilation
    let compile_id = Arc::new(AtomicU64::new(0));
    let is_compiling = Arc::new(AtomicBool::new(false));

    // Initial compilation (async)
    spawn_compile(&path, format, run_after, parser_backend, Arc::clone(&compile_id), Arc::clone(&is_compiling));

    // Set up file watcher
    let (tx, rx) = channel();

    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                let _ = tx.send(event);
            }
        },
        Config::default().with_poll_interval(Duration::from_millis(100)),
    )
    .expect("Failed to create file watcher");

    watcher
        .watch(&watch_path, RecursiveMode::Recursive)
        .expect("Failed to watch path");

    // Debounce tracking
    let mut last_event = Instant::now();
    let debounce_duration = Duration::from_millis(200);

    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(event) => {
                // Check if this is a relevant .rs file change
                let is_rust_file = event
                    .paths
                    .iter()
                    .any(|p| p.extension().map(|e| e == "rs").unwrap_or(false));

                if is_rust_file {
                    last_event = Instant::now();
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Check if we should trigger a compile (debounced)
                if last_event.elapsed() > debounce_duration && last_event.elapsed() < debounce_duration + Duration::from_millis(100) {
                    // Cancel any in-flight compilation by incrementing the ID
                    compile_id.fetch_add(1, Ordering::SeqCst);

                    if matches!(format, OutputFormat::Human) {
                        if is_compiling.load(Ordering::SeqCst) {
                            println!("\n{} Cancelling previous compile...", "⟳".yellow());
                        }
                        println!("\n{}", "─".repeat(50).dimmed());
                        println!("{} Change detected\n", "⟳".cyan().bold());
                    }

                    spawn_compile(&path, format, run_after, parser_backend, Arc::clone(&compile_id), Arc::clone(&is_compiling));
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
}

fn spawn_compile(
    path: &PathBuf,
    format: OutputFormat,
    run_after: bool,
    parser_backend: ParserBackend,
    compile_id: Arc<std::sync::atomic::AtomicU64>,
    is_compiling: Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::Ordering;

    let path = path.clone();
    let my_id = compile_id.load(Ordering::SeqCst);

    std::thread::spawn(move || {
        is_compiling.store(true, Ordering::SeqCst);

        let start = Instant::now();

        // Check if we've been cancelled
        if compile_id.load(Ordering::SeqCst) != my_id {
            is_compiling.store(false, Ordering::SeqCst);
            return;
        }

        let krate = match parse_with_backend(&path, parser_backend, format) {
            Ok(k) => k,
            Err(e) => {
                if compile_id.load(Ordering::SeqCst) == my_id {
                    if matches!(format, OutputFormat::Human) {
                        eprintln!("{} {}", "Parse error:".red().bold(), e);
                    }
                }
                is_compiling.store(false, Ordering::SeqCst);
                return;
            }
        };

        // Check cancellation again
        if compile_id.load(Ordering::SeqCst) != my_id {
            is_compiling.store(false, Ordering::SeqCst);
            return;
        }

        // Type checking
        let registry = Arc::new(TypeRegistry::new());
        registry.init_from_crate(&krate);
        let type_ctx = crate::typeck::TypeContext::new(Arc::clone(&registry));
        let mut type_checker = crate::typeck::TypeChecker::new(&type_ctx, &krate);

        if let Err(e) = type_checker.check_crate() {
            if compile_id.load(Ordering::SeqCst) == my_id {
                if matches!(format, OutputFormat::Human) {
                    eprintln!("{} {}", "Type error:".red().bold(), e);
                }
            }
            is_compiling.store(false, Ordering::SeqCst);
            return;
        }

        if type_ctx.has_errors() {
            if compile_id.load(Ordering::SeqCst) == my_id {
                for diag in type_ctx.take_diagnostics() {
                    if matches!(format, OutputFormat::Human) {
                        eprintln!("{} {}", "Error:".red().bold(), diag.message);
                    }
                }
            }
            is_compiling.store(false, Ordering::SeqCst);
            return;
        }

        let compile_time = start.elapsed();

        // Final cancellation check before output
        if compile_id.load(Ordering::SeqCst) != my_id {
            is_compiling.store(false, Ordering::SeqCst);
            return;
        }

        if matches!(format, OutputFormat::Human) {
            println!(
                "{} {} in {:.2}ms",
                "⚡".yellow(),
                "Compiled".green().bold(),
                compile_time.as_secs_f64() * 1000.0
            );
        }

        if run_after && compile_id.load(Ordering::SeqCst) == my_id {
            let mut codegen = match crate::codegen::CodeGenerator::new(Arc::clone(&registry)) {
                Ok(c) => c,
                Err(e) => {
                    if matches!(format, OutputFormat::Human) {
                        eprintln!("{} {}", "Codegen error:".red().bold(), e);
                    }
                    is_compiling.store(false, Ordering::SeqCst);
                    return;
                }
            };
            codegen.compile_crate(&krate);

            if matches!(format, OutputFormat::Human) {
                println!("{} {}", "⚡".yellow(), "Running...".cyan());
            }

            match codegen.run_main() {
                Ok(exit_code) => {
                    if matches!(format, OutputFormat::Human) {
                        println!("{} {} with code {}\n", "⚡".yellow(), "Finished".green().bold(), exit_code);
                    }
                }
                Err(e) => {
                    if matches!(format, OutputFormat::Human) {
                        eprintln!("{} {}", "Runtime error:".red().bold(), e);
                    }
                }
            }
        }

        is_compiling.store(false, Ordering::SeqCst);
    });
}
