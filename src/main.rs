fn main() {
    // Increase stack size to 16MB for Chumsky parser construction
    // The default 1MB stack overflows when building complex recursive parsers
    const STACK_SIZE: usize = 16 * 1024 * 1024;

    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            bolt::cli::run_cli();
        })
        .expect("Failed to spawn thread with larger stack")
        .join()
        .expect("Thread panicked");
}
