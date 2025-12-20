//! Command-line interface for Bolt

use clap::{Parser as ClapParser, Subcommand};
use colored::Colorize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(ClapParser)]
#[command(name = "bolt")]
#[command(about = "Lightning-fast Rust compiler for development")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
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
    },

    /// Check a Rust file for errors (no codegen)
    Check {
        /// Path to Rust file or project
        path: PathBuf,
    },

    /// Show cache statistics
    Cache {
        #[command(subcommand)]
        action: CacheAction,
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
        Commands::Run { path, args } => {
            run_file(&path, &args);
        }
        Commands::Build { path, output, release } => {
            build_file(&path, output.as_deref(), release);
        }
        Commands::Check { path } => {
            check_file(&path);
        }
        Commands::Cache { action } => {
            handle_cache(action);
        }
    }
}

fn run_file(path: &PathBuf, _args: &[String]) {
    let start = Instant::now();

    println!("{} {}", "Compiling".green().bold(), path.display());

    let parser = crate::parser::Parser::new();
    let krate = match parser.parse_crate(path) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{}: {}", "Error".red().bold(), e);
            std::process::exit(1);
        }
    };

    let type_ctx = crate::typeck::TypeContext::new();
    let mut type_checker = crate::typeck::TypeChecker::new(&type_ctx, &krate);
    if let Err(e) = type_checker.check_crate() {
        eprintln!("{}: {}", "Type error".red().bold(), e);
        std::process::exit(1);
    }

    if type_ctx.has_errors() {
        eprintln!("{}: Type checking failed", "Error".red().bold());
        std::process::exit(1);
    }

    let borrow_checker = crate::borrowck::BorrowChecker::new();
    borrow_checker.check_crate(&krate);
    if borrow_checker.has_errors() {
        for diag in borrow_checker.take_diagnostics() {
            eprintln!("{}: {}", "Borrow error".red().bold(), diag.message);
        }
        std::process::exit(1);
    }

    let mut codegen = match crate::codegen::CodeGenerator::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}: {}", "Codegen error".red().bold(), e);
            std::process::exit(1);
        }
    };

    if let Err(e) = codegen.compile_crate(&krate) {
        eprintln!("{}: {}", "Compilation error".red().bold(), e);
        std::process::exit(1);
    }

    let compile_time = start.elapsed();
    println!(
        "{} in {:.2}ms",
        "Compiled".green().bold(),
        compile_time.as_secs_f64() * 1000.0
    );

    println!("{}", "Running...".cyan());
    println!();

    match codegen.run_main() {
        Ok(exit_code) => {
            println!();
            println!(
                "{} with code {}",
                "Finished".green().bold(),
                exit_code
            );
        }
        Err(e) => {
            eprintln!("{}: {}", "Runtime error".red().bold(), e);
            std::process::exit(1);
        }
    }
}

fn build_file(path: &PathBuf, _output: Option<&std::path::Path>, release: bool) {
    let start = Instant::now();

    let mode = if release { "release" } else { "debug" };
    println!(
        "{} {} [{}]",
        "Compiling".green().bold(),
        path.display(),
        mode
    );

    let parser = crate::parser::Parser::new();
    let krate = match parser.parse_crate(path) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{}: {}", "Error".red().bold(), e);
            std::process::exit(1);
        }
    };

    let type_ctx = crate::typeck::TypeContext::new();
    let mut type_checker = crate::typeck::TypeChecker::new(&type_ctx, &krate);
    if let Err(e) = type_checker.check_crate() {
        eprintln!("{}: {}", "Type error".red().bold(), e);
        std::process::exit(1);
    }

    let borrow_checker = crate::borrowck::BorrowChecker::new();
    borrow_checker.check_crate(&krate);
    if borrow_checker.has_errors() {
        for diag in borrow_checker.take_diagnostics() {
            eprintln!("{}: {}", "Borrow error".red().bold(), diag.message);
        }
        std::process::exit(1);
    }

    let mut codegen = match crate::codegen::CodeGenerator::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}: {}", "Codegen error".red().bold(), e);
            std::process::exit(1);
        }
    };

    if let Err(e) = codegen.compile_crate(&krate) {
        eprintln!("{}: {}", "Compilation error".red().bold(), e);
        std::process::exit(1);
    }

    let compile_time = start.elapsed();
    println!(
        "{} in {:.2}ms",
        "Finished".green().bold(),
        compile_time.as_secs_f64() * 1000.0
    );
}

fn check_file(path: &PathBuf) {
    let start = Instant::now();

    println!("{} {}", "Checking".cyan().bold(), path.display());

    let parser = crate::parser::Parser::new();
    let krate = match parser.parse_crate(path) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{}: {}", "Error".red().bold(), e);
            std::process::exit(1);
        }
    };

    let parse_time = start.elapsed();

    let type_start = Instant::now();
    let type_ctx = crate::typeck::TypeContext::new();
    let mut type_checker = crate::typeck::TypeChecker::new(&type_ctx, &krate);
    if let Err(e) = type_checker.check_crate() {
        eprintln!("{}: {}", "Type error".red().bold(), e);
        std::process::exit(1);
    }
    let type_time = type_start.elapsed();

    let borrow_start = Instant::now();
    let borrow_checker = crate::borrowck::BorrowChecker::new();
    borrow_checker.check_crate(&krate);
    let borrow_time = borrow_start.elapsed();

    let has_errors = type_ctx.has_errors() || borrow_checker.has_errors();

    if has_errors {
        for diag in borrow_checker.take_diagnostics() {
            eprintln!("{}: {}", "Error".red().bold(), diag.message);
        }
        std::process::exit(1);
    }

    let total_time = start.elapsed();

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
