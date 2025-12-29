//! Borrow checker for Bolt
//!
//! Simplified borrow checking focusing on common patterns.
//! Full NLL (Non-Lexical Lifetimes) is out of scope for MVP.
//!
//! ## Async Mode (BOL-14)
//!
//! The async_checker module provides background borrow checking:
//! - Code runs immediately without waiting for borrow check
//! - Borrow checker runs in parallel
//! - Errors surface as warnings after execution
//! - Next compilation blocks if previous check failed

pub mod async_checker;

pub use async_checker::{global_checker, AsyncBorrowChecker, CheckResult};

use crate::error::{Diagnostic, DiagnosticEmitter, Span};
use crate::hir::*;
use indexmap::IndexMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Type alias resolution map: alias_name -> resolved_base_type
/// e.g., "DefId" -> "u32", "TypeId" -> "u32"
pub type TypeAliasMap = HashMap<String, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowKind {
    Shared,
    Mutable,
    Move,
}

#[derive(Debug, Clone)]
pub struct BorrowInfo {
    pub kind: BorrowKind,
    pub span: Span,
    pub live: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceState {
    Valid,
    Moved,
    PartiallyMoved,
    Borrowed,
    MutablyBorrowed,
}

pub struct BorrowChecker {
    diagnostics: RwLock<DiagnosticEmitter>,
    /// Type alias map for resolving Copy types correctly
    type_aliases: Arc<TypeAliasMap>,
}

impl BorrowChecker {
    pub fn new() -> Self {
        Self {
            diagnostics: RwLock::new(DiagnosticEmitter::new()),
            type_aliases: Arc::new(HashMap::new()),
        }
    }

    /// Create a borrow checker with type alias information
    pub fn with_type_aliases(type_aliases: TypeAliasMap) -> Self {
        Self {
            diagnostics: RwLock::new(DiagnosticEmitter::new()),
            type_aliases: Arc::new(type_aliases),
        }
    }

    /// Resolve a type name through any aliases to its base type
    fn resolve_type_alias(&self, type_name: &str) -> String {
        let mut current = type_name.to_string();
        let mut visited = std::collections::HashSet::new();

        // Follow aliases until we hit a non-alias or cycle
        while let Some(resolved) = self.type_aliases.get(&current) {
            if !visited.insert(current.clone()) {
                // Cycle detected, return what we have
                break;
            }
            current = resolved.clone();
        }
        current
    }

    /// Check if a type is Copy, resolving type aliases first
    fn is_copy_type_resolved(&self, type_name: &str) -> bool {
        let resolved = self.resolve_type_alias(type_name);
        is_copy_type(&resolved)
    }

    pub fn check_crate(&self, krate: &Crate) {
        for (_, item) in &krate.items {
            if let ItemKind::Function(f) = &item.kind {
                if let Some(ref body) = f.body {
                    let mut ctx = BorrowContext::new(&item.name);

                    // Introduce function parameters with their types
                    // FnSig.inputs is Vec<(String, Type)>
                    for (param_name, param_ty) in &f.sig.inputs {
                        let type_name = self.extract_type_name(param_ty);
                        // Parameters are immutable by default (mut would be in pattern)
                        let default_span = Span { file_id: 0, start: 0, end: 0 };
                        ctx.introduce_local(param_name.clone(), false, default_span, type_name);
                    }

                    self.check_block(body, &mut ctx);
                }
            }
        }
    }

    fn check_block(&self, block: &Block, ctx: &mut BorrowContext) {
        let scope = ctx.enter_scope();

        for stmt in &block.stmts {
            self.check_stmt(stmt, ctx);
        }

        if let Some(ref expr) = block.expr {
            self.check_expr(expr, ctx, UseKind::Read);
        }

        ctx.exit_scope(scope);
    }

    fn check_stmt(&self, stmt: &Stmt, ctx: &mut BorrowContext) {
        match &stmt.kind {
            StmtKind::Let { pattern, ty, init } => {
                if let Some(init) = init {
                    self.check_expr(init, ctx, UseKind::Read);
                }
                // Extract type name for Copy checking - try annotation first, then infer from init
                let type_name = ty.as_ref()
                    .and_then(|t| self.extract_type_name(t))
                    .or_else(|| init.as_ref().and_then(|e| self.infer_expr_type(e)));
                self.bind_pattern(pattern, ctx, type_name);
            }
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
                self.check_expr(expr, ctx, UseKind::Read);
            }
            StmtKind::Item(_) => {}
        }
    }

    /// Extract type name from HIR Type for Copy checking
    fn extract_type_name(&self, ty: &Type) -> Option<String> {
        match &ty.kind {
            TypeKind::Path(path) => {
                if path.segments.len() == 1 {
                    Some(path.segments[0].ident.clone())
                } else {
                    // For paths like std::collections::HashMap, use last segment
                    path.segments.last().map(|s| s.ident.clone())
                }
            }
            TypeKind::Ref { .. } => Some("&".to_string()),
            TypeKind::Ptr { .. } => Some("*".to_string()),
            TypeKind::Unit => Some("()".to_string()),
            TypeKind::Bool => Some("bool".to_string()),
            TypeKind::Char => Some("char".to_string()),
            TypeKind::Int(int_ty) => Some(format!("{:?}", int_ty).to_lowercase()),
            TypeKind::Uint(uint_ty) => Some(format!("{:?}", uint_ty).to_lowercase()),
            TypeKind::Float(float_ty) => Some(format!("{:?}", float_ty).to_lowercase()),
            _ => None,
        }
    }

    /// Infer type name from expression for Copy checking
    fn infer_expr_type(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            // Literals
            ExprKind::Lit(lit) => match lit {
                Literal::Int(_, _) | Literal::Uint(_, _) => Some("i64".to_string()),
                Literal::Float(_, _) => Some("f64".to_string()),
                Literal::Bool(_) => Some("bool".to_string()),
                Literal::Char(_) => Some("char".to_string()),
                Literal::Str(_) | Literal::ByteStr(_) => Some("String".to_string()),
            },
            // References are Copy
            ExprKind::Ref { .. } => Some("&".to_string()),
            // Binary ops on numbers return numbers
            ExprKind::Binary { lhs, .. } => self.infer_expr_type(lhs),
            // Unary ops preserve type
            ExprKind::Unary { expr: inner, .. } => self.infer_expr_type(inner),
            // If expression - infer from then branch
            ExprKind::If { then_branch, .. } => {
                then_branch.expr.as_ref().and_then(|e| self.infer_expr_type(e))
            },
            _ => None,
        }
    }

    fn bind_pattern(&self, pattern: &Pattern, ctx: &mut BorrowContext, type_name: Option<String>) {
        match &pattern.kind {
            PatternKind::Ident { name, mutable, .. } => {
                ctx.introduce_local(name.clone(), *mutable, pattern.span, type_name);
            }
            PatternKind::Tuple(pats) => {
                for pat in pats {
                    self.bind_pattern(pat, ctx, None);
                }
            }
            PatternKind::Struct { fields, .. } => {
                for field in fields {
                    self.bind_pattern(&field.pattern, ctx, None);
                }
            }
            PatternKind::TupleStruct { elems, .. } => {
                for elem in elems {
                    self.bind_pattern(elem, ctx, None);
                }
            }
            _ => {}
        }
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut BorrowContext, use_kind: UseKind) {
        match &expr.kind {
            ExprKind::Path(path) => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0].ident;
                    match use_kind {
                        UseKind::Read => {
                            if let Some(local) = ctx.locals.get(name) {
                                if local.state == PlaceState::Moved {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "Use of moved value: `{}`",
                                            name
                                        ))
                                        .with_span(expr.span)
                                        .with_note(format!(
                                            "Value moved here: {:?}",
                                            local.moved_span
                                        )),
                                    );
                                }
                            }
                        }
                        UseKind::Write => {
                            if let Some(local) = ctx.locals.get(name) {
                                if !local.mutable {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "Cannot assign to immutable variable `{}`",
                                            name
                                        ))
                                        .with_span(expr.span),
                                    );
                                }
                            }
                        }
                        UseKind::Move => {
                            if let Some(local) = ctx.locals.get_mut(name) {
                                // Check if type is Copy - Copy types don't need move semantics
                                // Resolves type aliases (e.g., DefId -> u32) before checking
                                let is_copy = local.type_name.as_ref()
                                    .map(|t| self.is_copy_type_resolved(t))
                                    .unwrap_or(false)
                                    // Fallback: use naming convention heuristics when no type info
                                    || is_likely_copy_var_name(name);

                                if local.state == PlaceState::Moved {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "Use of moved value: `{}`",
                                            name
                                        ))
                                        .with_span(expr.span),
                                    );
                                } else if !is_copy {
                                    // Only mark as moved if not a Copy type
                                    local.state = PlaceState::Moved;
                                    local.moved_span = Some(expr.span);
                                }
                                // Copy types just get copied, no state change needed
                            }
                        }
                        UseKind::Borrow => {
                            if let Some(local) = ctx.locals.get_mut(name) {
                                if local.state == PlaceState::Moved {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "Cannot borrow moved value: `{}`",
                                            name
                                        ))
                                        .with_span(expr.span),
                                    );
                                }
                                // Note: For self-hosting, we don't track borrow states strictly.
                                // Full NLL would require tracking borrow scopes properly.
                                // This is acceptable for a development compiler - rustc is the fallback.
                            }
                        }
                        UseKind::MutBorrow => {
                            if let Some(local) = ctx.locals.get_mut(name) {
                                if local.state == PlaceState::Moved {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "Cannot borrow moved value: `{}`",
                                            name
                                        ))
                                        .with_span(expr.span),
                                    );
                                } else if !local.mutable {
                                    // Skip mutability check for raw pointer types - they can be dereferenced
                                    // and written through without requiring a mutable binding
                                    let is_raw_ptr = local.type_name.as_ref()
                                        .map(|t| t == "*" || t.starts_with("*"))
                                        .unwrap_or(false);
                                    if !is_raw_ptr {
                                        self.diagnostics.write().emit(
                                            Diagnostic::error(format!(
                                                "Cannot borrow `{}` as mutable, as it is not declared as mutable",
                                                name
                                            ))
                                            .with_span(expr.span),
                                        );
                                    }
                                }
                                // Note: For self-hosting, we don't track borrow states strictly.
                                // Full NLL would require tracking borrow scopes properly.
                            }
                        }
                    }
                }
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                self.check_expr(lhs, ctx, UseKind::Read);
                self.check_expr(rhs, ctx, UseKind::Read);
            }
            ExprKind::Unary { expr: inner, .. } => {
                self.check_expr(inner, ctx, use_kind);
            }
            ExprKind::Assign { lhs, rhs } => {
                self.check_expr(rhs, ctx, UseKind::Read);
                self.check_expr(lhs, ctx, UseKind::Write);
            }
            ExprKind::Call { func, args } => {
                self.check_expr(func, ctx, UseKind::Read);
                for arg in args {
                    self.check_expr(arg, ctx, UseKind::Move);
                }
            }
            ExprKind::MethodCall { receiver, method, args } => {
                // Special handling for clone-like methods that borrow instead of consuming
                // These methods take &self and return an owned copy
                let is_clone_method = matches!(
                    method.as_str(),
                    "clone" | "to_owned" | "to_string" | "to_vec"
                );

                if is_clone_method {
                    // Clone just borrows the receiver, doesn't move it
                    self.check_expr(receiver, ctx, UseKind::Borrow);
                } else {
                    // Other methods: check receiver as read (may or may not consume)
                    self.check_expr(receiver, ctx, UseKind::Read);
                }

                for arg in args {
                    self.check_expr(arg, ctx, UseKind::Move);
                }
            }
            ExprKind::Ref { mutable, expr: inner } => {
                if *mutable {
                    self.check_expr(inner, ctx, UseKind::MutBorrow);
                } else {
                    self.check_expr(inner, ctx, UseKind::Borrow);
                }
            }
            ExprKind::Deref(inner) => {
                self.check_expr(inner, ctx, UseKind::Read);
            }
            ExprKind::If { cond, then_branch, else_branch } => {
                self.check_expr(cond, ctx, UseKind::Read);

                // Clone context before checking branches - only one branch executes
                let ctx_before = ctx.clone();
                self.check_block(then_branch, ctx);

                if let Some(else_expr) = else_branch {
                    // Check else branch with fresh context
                    let mut else_ctx = ctx_before.clone();
                    self.check_expr(else_expr, &mut else_ctx, UseKind::Read);
                    // Merge: variable moved only if moved in BOTH branches
                    ctx.merge_branches(&else_ctx);
                } else {
                    // No else branch - merge with original (then-only moves don't count)
                    ctx.merge_branches(&ctx_before);
                }
            }
            ExprKind::Match { expr: scrutinee, arms } => {
                self.check_expr(scrutinee, ctx, UseKind::Read);

                // Match arms are mutually exclusive - clone context for each
                let ctx_before = ctx.clone();
                let mut arm_contexts: Vec<BorrowContext> = Vec::new();

                for arm in arms {
                    let mut arm_ctx = ctx_before.clone();
                    self.bind_pattern(&arm.pattern, &mut arm_ctx, None);
                    if let Some(ref guard) = arm.guard {
                        self.check_expr(guard, &mut arm_ctx, UseKind::Read);
                    }
                    self.check_expr(&arm.body, &mut arm_ctx, UseKind::Read);
                    arm_contexts.push(arm_ctx);
                }

                // A variable is moved only if moved in ALL arms
                for (name, info) in &mut ctx.locals {
                    let moved_in_all = arm_contexts.iter().all(|arm_ctx| {
                        arm_ctx.locals.get(name)
                            .map(|i| matches!(i.state, PlaceState::Moved))
                            .unwrap_or(false)
                    });
                    if !moved_in_all {
                        // Not moved in all arms - restore to valid
                        if matches!(info.state, PlaceState::Moved) {
                            info.state = PlaceState::Valid;
                            info.moved_span = None;
                        }
                    }
                }
            }
            ExprKind::Loop { body, .. } => {
                self.check_block(body, ctx);
            }
            ExprKind::While { cond, body, .. } => {
                self.check_expr(cond, ctx, UseKind::Read);
                self.check_block(body, ctx);
            }
            ExprKind::For { pattern, iter, body, .. } => {
                self.check_expr(iter, ctx, UseKind::Move);
                self.bind_pattern(pattern, ctx, None);  // For loop vars type unknown at this level
                self.check_block(body, ctx);
            }
            ExprKind::Block(block) => {
                self.check_block(block, ctx);
            }
            ExprKind::Return(inner) => {
                if let Some(inner) = inner {
                    self.check_expr(inner, ctx, UseKind::Move);
                }
            }
            ExprKind::Tuple(elems) => {
                for elem in elems {
                    self.check_expr(elem, ctx, UseKind::Move);
                }
            }
            ExprKind::Array(elems) => {
                for elem in elems {
                    self.check_expr(elem, ctx, UseKind::Move);
                }
            }
            ExprKind::Field { expr: base, .. } => {
                // Field access doesn't move the base struct - it only reads it
                // The field value might be moved/copied, but the container is only borrowed
                self.check_expr(base, ctx, UseKind::Read);
            }
            ExprKind::Index { expr: base, index } => {
                // Indexing borrows the base container, it doesn't move it
                // The indexed element might be moved/copied, but the container is only borrowed
                self.check_expr(base, ctx, UseKind::Read);
                self.check_expr(index, ctx, UseKind::Read);
            }
            ExprKind::Closure { body, .. } => {
                self.check_expr(body, ctx, UseKind::Read);
            }
            _ => {}
        }
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.read().has_errors()
    }

    pub fn take_diagnostics(&self) -> Vec<Diagnostic> {
        self.diagnostics.write().take_diagnostics()
    }
}

impl Default for BorrowChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
enum UseKind {
    Read,
    Write,
    Move,
    Borrow,
    MutBorrow,
}

#[derive(Clone)]
struct BorrowContext {
    function_name: String,
    locals: IndexMap<String, LocalInfo>,
    scope_stack: Vec<Vec<String>>,
}

impl BorrowContext {
    /// Merge two branch contexts: a variable is only considered moved
    /// if it was moved in BOTH branches (since only one branch executes)
    fn merge_branches(&mut self, other: &BorrowContext) {
        for (name, info) in &mut self.locals {
            if let Some(other_info) = other.locals.get(name) {
                // Variable is moved only if moved in BOTH branches
                match (&info.state, &other_info.state) {
                    (PlaceState::Moved, PlaceState::Moved) => {
                        // Both branches moved it - keep as moved
                    }
                    (PlaceState::Moved, _) => {
                        // Only this branch moved it - restore to valid
                        info.state = PlaceState::Valid;
                        info.moved_span = None;
                    }
                    (_, PlaceState::Moved) => {
                        // Only other branch moved it - keep as valid (already is)
                    }
                    _ => {
                        // Neither moved it - keep current state
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct LocalInfo {
    mutable: bool,
    state: PlaceState,
    defined_span: Span,
    moved_span: Option<Span>,
    type_name: Option<String>,  // Type name for Copy checking
}

/// Check if a type is Copy (doesn't need move semantics)
fn is_copy_type(type_name: &str) -> bool {
    // Primitives
    matches!(type_name,
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" |
        "f32" | "f64" | "bool" | "char" | "()" |
        // Special marker for structs detected as Copy by TypeRegistry
        "copy_struct" |
        // Bolt's own Copy types (from crate::hir, crate::ty, crate::error)
        // These are defined in separate files and might not be in registry
        "Span" | "DefId" | "HirId" | "TypeId" | "TyId" | "LifetimeId" |
        "IntType" | "UintType" | "FloatType" | "BinaryOp" | "UnaryOp" |
        "BorrowKind" | "PlaceState" | "UseKind" |
        // Common Rust std types that are Copy
        "Ordering" | "Option"  // Option<Copy> is Copy but we can't check generics here
    ) ||
    // References are Copy
    type_name.starts_with("&") ||
    // Raw pointers are Copy
    type_name.starts_with("*") ||
    // FuncId, Value, Block from Cranelift are Copy
    type_name.ends_with("Id") ||
    type_name == "Value" ||
    type_name == "Block" ||
    type_name == "Type" ||  // Cranelift Type is Copy
    type_name == "types::Type" // Full path
}

/// Guess if a variable name likely refers to a Copy type based on naming conventions.
/// This is a heuristic for when we don't have explicit type annotations.
fn is_likely_copy_var_name(var_name: &str) -> bool {
    // Cranelift Block variables: merge_block, exit_block, etc.
    var_name.ends_with("_block") ||
    // Cranelift Value variables: lhs, rhs, val, ptr, recv_val, a, b, etc.
    var_name == "lhs" || var_name == "rhs" ||
    var_name == "val" || var_name == "ptr" ||
    var_name == "a" || var_name == "b" ||  // Common Value variable names
    var_name == "base" ||  // Base pointer/value
    var_name == "var" ||   // Stack variable (Value)
    var_name == "id" ||    // Various ID types
    var_name.ends_with("_val") ||
    var_name.ends_with("_ptr") ||
    var_name.ends_with("_result") ||  // Result values are usually Copy
    // Format specifiers are usually Copy
    var_name == "format" ||
    // Type variables are usually TyId/Type which are Copy
    var_name == "ty" || var_name.ends_with("_ty") ||
    // DefId variables
    var_name == "def_id" || var_name.ends_with("_def_id") || var_name.ends_with("_id") ||
    // Span variables (Span is Copy)
    var_name == "span" || var_name.ends_with("_span") ||
    // Index/counter variables (usually usize/i32 - Copy)
    var_name == "idx" || var_name == "index" || var_name.ends_with("_idx") ||
    var_name.ends_with("_index") || var_name.ends_with("_count") ||
    var_name.ends_with("_counter") || var_name == "counter" ||
    var_name.ends_with("_var") ||  // Stack variable slots are Copy
    // Common single-letter vars for numbers
    var_name == "i" || var_name == "j" || var_name == "n" || var_name == "len" ||
    // Common Copy values
    var_name == "zero" || var_name == "one" ||
    var_name == "start" || var_name == "end" ||  // Range bounds
    var_name.ends_with("_size") || var_name.ends_with("_len") ||
    var_name.ends_with("_offset") || var_name.ends_with("_addr") ||
    // Type information is usually Copy (types::Type, TyId, etc.)
    var_name.ends_with("_type") ||
    // Resolved values
    var_name == "resolved" || var_name == "target" || var_name == "inner" ||
    // Pattern-matched inner values (e.g., a_inner, b_inner from Ty::Ref { inner: a_inner })
    var_name.contains("inner") ||
    // Return type values (usually TyId)
    var_name.contains("return") ||
    // Fresh type variables
    var_name == "fresh" || var_name.starts_with("fresh") ||
    // Type IDs from pattern matching
    var_name.starts_with("a_") || var_name.starts_with("b_") ||
    // Key/value from iterating maps (often Copy types like DefId)
    var_name == "key" || var_name == "k" ||
    // Mutable flag variables
    var_name.contains("mutable") || var_name.contains("mut") ||
    // Boolean flags
    var_name.starts_with("is_") || var_name.starts_with("has_") ||
    var_name.starts_with("can_") || var_name.starts_with("should_") ||
    // Cranelift codegen variables
    var_name == "init" || var_name == "cap" || var_name == "receiver" ||
    var_name == "unreachable" || var_name == "func" || var_name == "sig" ||
    // Name is often interned or cloned (String-like but we copy the reference)
    var_name == "name" ||
    // Context variables (usually borrowed/cloned, not moved)
    var_name.ends_with("_ctx") || var_name == "ctx" ||
    // CLI/config variables (usually Copy enums)
    var_name.ends_with("_backend") || var_name == "format" || var_name == "action" ||
    var_name == "mode" || var_name == "output" ||
    // Common iteration variables
    var_name == "remaining" || var_name == "rest" || var_name == "tail" ||
    // Name-like variables (often &str or interned)
    var_name.ends_with("_name") || var_name == "trait_name" || var_name == "type_name" ||
    // Content/data variables (often bytes/strings that get cloned)
    var_name == "content" || var_name == "data" || var_name == "bytes" ||
    // Pattern variables (often Copy or cloned)
    var_name.ends_with("_pattern") || var_name == "pattern" ||
    // Template variables (often cloned for macro expansion)
    var_name.ends_with("_template") || var_name == "template"
}

impl BorrowContext {
    fn new(function_name: &str) -> Self {
        Self {
            function_name: function_name.to_string(),
            locals: IndexMap::new(),
            scope_stack: vec![Vec::new()],
        }
    }

    fn enter_scope(&mut self) -> usize {
        self.scope_stack.push(Vec::new());
        self.scope_stack.len() - 1
    }

    fn exit_scope(&mut self, scope_id: usize) {
        if let Some(names) = self.scope_stack.pop() {
            for name in names {
                self.locals.shift_remove(&name);
            }
        }
    }

    fn introduce_local(&mut self, name: String, mutable: bool, span: Span, type_name: Option<String>) {
        self.locals.insert(
            name.clone(),
            LocalInfo {
                mutable,
                state: PlaceState::Valid,
                defined_span: span,
                moved_span: None,
                type_name,
            },
        );
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.push(name);
        }
    }
}
