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
#[command(about = "Lightning-fast Rust compiler for development")]
#[command(version)]
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
    },

    /// Check a Rust file for errors (no codegen)
    Check {
        /// Path to Rust file or project
        path: PathBuf,

        /// Output format for errors/diagnostics
        #[arg(long, short = 'f', value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,
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
        Commands::Run { path, args, format, sync_borrow } => {
            run_file(&path, &args, format, !sync_borrow);
        }
        Commands::Build { path, output, release, format } => {
            build_file(&path, output.as_deref(), release, format);
        }
        Commands::Check { path, format } => {
            check_file(&path, format);
        }
        Commands::Cache { action } => {
            handle_cache(action);
        }
        Commands::Watch { path, format, run } => {
            watch_file(&path, format, run);
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

fn run_file(path: &PathBuf, _args: &[String], format: OutputFormat, async_mode: bool) {
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
            emit_diagnostics(format, json_diags);
            std::process::exit(1);
        }
    }

    if matches!(format, OutputFormat::Human) {
        println!("{} {}", "Compiling".green().bold(), path.display());
    }

    let parser = crate::parser::Parser::new();
    let krate = match parser.parse_crate(path) {
        Ok(k) => k,
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::UnexpectedToken, &e.to_string(), Some(path), None, None);
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

    if async_mode {
        // Spawn borrow check in background - don't wait for it
        let checker = crate::borrowck::global_checker();
        checker.spawn_check(path.clone(), Arc::clone(&krate_arc));
    } else {
        // Synchronous borrow checking (traditional mode)
        let borrow_checker = crate::borrowck::BorrowChecker::new();
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
            emit_diagnostics(format, diagnostics);
            std::process::exit(1);
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
            "{} in {:.2}ms{}",
            "Compiled".green().bold(),
            compile_time.as_secs_f64() * 1000.0,
            if async_mode { " (borrow check async)" } else { "" }
        );
        println!("{}", "Running...".cyan());
        println!();
    }

    match codegen.run_main() {
        Ok(exit_code) => {
            if matches!(format, OutputFormat::Human) {
                println!();
                println!(
                    "{} with code {}",
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

fn build_file(path: &PathBuf, _output: Option<&std::path::Path>, release: bool, format: OutputFormat) {
    let start = Instant::now();

    let mode = if release { "release" } else { "debug" };
    if matches!(format, OutputFormat::Human) {
        println!(
            "{} {} [{}]",
            "Compiling".green().bold(),
            path.display(),
            mode
        );
    }

    let parser = crate::parser::Parser::new();
    let krate = match parser.parse_crate(path) {
        Ok(k) => k,
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::UnexpectedToken, &e.to_string(), Some(path), None, None);
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

    let borrow_checker = crate::borrowck::BorrowChecker::new();
    borrow_checker.check_crate(&krate);
    if borrow_checker.has_errors() {
        let diagnostics: Vec<JsonDiagnostic> = borrow_checker
            .take_diagnostics()
            .into_iter()
            .map(|d| {
                JsonDiagnostic::error(ErrorCode::BorrowOfMovedValue, &d.message)
                    .with_notes_from_vec(d.notes)
            })
            .collect();
        emit_diagnostics(format, diagnostics);
        std::process::exit(1);
    }

    let mut codegen = match crate::codegen::CodeGenerator::new(Arc::clone(&registry)) {
        Ok(c) => c,
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::InternalError, &e.to_string(), None, None, None);
        }
    };

    if let Err(e) = codegen.compile_crate(&krate) {
        emit_error_and_exit(format, ErrorCode::InternalError, &e.to_string(), Some(path), None, None);
    }

    let compile_time = start.elapsed();

    if matches!(format, OutputFormat::Human) {
        println!(
            "{} in {:.2}ms",
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

fn check_file(path: &PathBuf, format: OutputFormat) {
    let start = Instant::now();

    if matches!(format, OutputFormat::Human) {
        println!("{} {}", "Checking".cyan().bold(), path.display());
    }

    let parser = crate::parser::Parser::new();
    let krate = match parser.parse_crate(path) {
        Ok(k) => k,
        Err(e) => {
            emit_error_and_exit(format, ErrorCode::UnexpectedToken, &e.to_string(), Some(path), None, None);
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
        emit_error_and_exit(format, ErrorCode::TypeMismatch, &e.to_string(), Some(path), None, None);
    }
    let type_time = type_start.elapsed();

    let borrow_start = Instant::now();
    let borrow_checker = crate::borrowck::BorrowChecker::new();
    borrow_checker.check_crate(&krate);
    let borrow_time = borrow_start.elapsed();

    let has_errors = type_ctx.has_errors() || borrow_checker.has_errors();

    if has_errors {
        let diagnostics: Vec<JsonDiagnostic> = borrow_checker
            .take_diagnostics()
            .into_iter()
            .map(|d| {
                JsonDiagnostic::error(ErrorCode::BorrowOfMovedValue, &d.message)
                    .with_notes_from_vec(d.notes)
            })
            .collect();
        emit_diagnostics(format, diagnostics);
        std::process::exit(1);
    }

    let total_time = start.elapsed();

    if matches!(format, OutputFormat::Human) {
        println!();
        println!("{}", "No errors found!".green().bold());
        println!();
        println!("  Parse:  {:>6.2}ms", parse_time.as_secs_f64() * 1000.0);
        println!("  Type:   {:>6.2}ms", type_time.as_secs_f64() * 1000.0);
        println!("  Borrow: {:>6.2}ms", borrow_time.as_secs_f64() * 1000.0);
        println!("  ──────────────────");
        println!(
            "  Total:  {:>6.2}ms",
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
            println!("{}", "Cache Statistics".cyan().bold());
            println!();
            println!("  Entries:    {}", stats.total_entries);
            println!("  Valid:      {}", stats.valid_entries);
            println!("  Hit rate:   {:.1}%", stats.hit_rate() * 100.0);
            println!("  Total size: {} KB", stats.total_size_bytes / 1024);
        }
        CacheAction::Clear => {
            match cache.clear() {
                Ok(()) => println!("{}", "Cache cleared".green().bold()),
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
                        "{} {} stale entries",
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

fn watch_file(path: &PathBuf, format: OutputFormat, run_after: bool) {
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
            "{} {} for changes...",
            "Watching".cyan().bold(),
            watch_path.display()
        );
        println!("  Press Ctrl+C to stop");
        println!("  Compilation runs in background - system stays responsive\n");
    }

    // Shared state for async compilation
    let compile_id = Arc::new(AtomicU64::new(0));
    let is_compiling = Arc::new(AtomicBool::new(false));

    // Initial compilation (async)
    spawn_compile(&path, format, run_after, Arc::clone(&compile_id), Arc::clone(&is_compiling));

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

                    spawn_compile(&path, format, run_after, Arc::clone(&compile_id), Arc::clone(&is_compiling));
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

        let parser = crate::parser::Parser::new();
        let krate = match parser.parse_crate(&path) {
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
                "{} in {:.2}ms",
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
                println!("{}", "Running...".dimmed());
            }

            match codegen.run_main() {
                Ok(exit_code) => {
                    if matches!(format, OutputFormat::Human) {
                        println!("{} with code {}\n", "Finished".green(), exit_code);
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
