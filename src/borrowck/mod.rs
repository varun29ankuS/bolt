//! Borrow checker for Bolt
//!
//! Implements ownership and borrowing rules:
//! - Move semantics for non-Copy types
//! - Borrow scope tracking (when references expire)
//! - Partial move tracking for struct fields
//! - Mutable borrow exclusivity
//!
//! ## Ownership Ledger (Blockchain-inspired)
//!
//! The ledger module provides an append-only log of all ownership events:
//! - Every create, move, borrow, drop is recorded
//! - Full audit trail for debugging
//! - LLM-friendly JSON output
//! - Visual history for error messages
//!
//! ## Async Mode
//!
//! The async_checker module provides background borrow checking:
//! - Code runs immediately without waiting for borrow check
//! - Borrow checker runs in parallel
//! - Errors surface as warnings after execution
//! - Next compilation blocks if previous check failed

pub mod async_checker;
pub mod ledger;
pub mod nll;

pub use async_checker::{global_checker, AsyncBorrowChecker, CheckResult};
pub use nll::NllChecker;

use crate::error::{Diagnostic, DiagnosticEmitter, Span};
use crate::hir::*;
use indexmap::IndexMap;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Type alias resolution map: alias_name -> resolved_base_type
/// e.g., "DefId" -> "u32", "TypeId" -> "u32"
pub type TypeAliasMap = HashMap<String, String>;

/// A place is a path to a memory location: `x`, `x.field`, `x.0`, `(*x).field`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Place {
    /// Root variable name
    pub base: String,
    /// Field projections (field names or tuple indices as strings)
    pub projections: Vec<String>,
}

impl Place {
    pub fn new(base: String) -> Self {
        Self { base, projections: vec![] }
    }

    pub fn with_field(mut self, field: String) -> Self {
        self.projections.push(field);
        self
    }

    /// Check if this place is a prefix of another (for partial moves)
    pub fn is_prefix_of(&self, other: &Place) -> bool {
        if self.base != other.base {
            return false;
        }
        if self.projections.len() > other.projections.len() {
            return false;
        }
        self.projections.iter().zip(&other.projections).all(|(a, b)| a == b)
    }

    pub fn display(&self) -> String {
        if self.projections.is_empty() {
            self.base.clone()
        } else {
            format!("{}.{}", self.base, self.projections.join("."))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowKind {
    Shared,
    Mutable,
}

/// An active borrow of a place
#[derive(Debug, Clone)]
pub struct Borrow {
    pub place: Place,
    pub kind: BorrowKind,
    pub span: Span,
    /// Scope ID where this borrow was created
    pub scope_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceState {
    Valid,
    Moved,
    PartiallyMoved,
}

pub struct BorrowChecker {
    diagnostics: RwLock<DiagnosticEmitter>,
    /// Type alias map for resolving Copy types correctly
    type_aliases: Arc<TypeAliasMap>,
    /// Set of type names known to be Copy
    copy_types: Arc<HashSet<String>>,
}

impl BorrowChecker {
    pub fn new() -> Self {
        Self {
            diagnostics: RwLock::new(DiagnosticEmitter::new()),
            type_aliases: Arc::new(HashMap::new()),
            copy_types: Arc::new(Self::default_copy_types()),
        }
    }

    /// Create a borrow checker with type alias information
    pub fn with_type_aliases(type_aliases: TypeAliasMap) -> Self {
        Self {
            diagnostics: RwLock::new(DiagnosticEmitter::new()),
            type_aliases: Arc::new(type_aliases),
            copy_types: Arc::new(Self::default_copy_types()),
        }
    }

    /// Create a borrow checker with full type information
    pub fn with_type_info(type_aliases: TypeAliasMap, copy_types: HashSet<String>) -> Self {
        let mut all_copy = Self::default_copy_types();
        all_copy.extend(copy_types);
        Self {
            diagnostics: RwLock::new(DiagnosticEmitter::new()),
            type_aliases: Arc::new(type_aliases),
            copy_types: Arc::new(all_copy),
        }
    }

    /// Default set of Copy types (primitives + common types)
    fn default_copy_types() -> HashSet<String> {
        let mut copy = HashSet::new();
        // Primitives
        for t in &["i8", "i16", "i32", "i64", "i128", "isize",
                   "u8", "u16", "u32", "u64", "u128", "usize",
                   "f32", "f64", "bool", "char", "()", "!"] {
            copy.insert(t.to_string());
        }
        // Bolt's internal Copy types
        for t in &["Span", "DefId", "HirId", "TypeId", "TyId", "LifetimeId",
                   "IntType", "UintType", "FloatType", "BinaryOp", "UnaryOp",
                   "BorrowKind", "PlaceState", "EventType", "ValueState", "Location",
                   "Ordering"] {
            copy.insert(t.to_string());
        }
        copy
    }

    /// Resolve a type name through any aliases to its base type
    fn resolve_type_alias(&self, type_name: &str) -> String {
        let mut current = type_name.to_string();
        let mut visited = HashSet::new();

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

    /// Check if a type is Copy
    fn is_copy_type(&self, type_name: &str) -> bool {
        // First resolve any aliases
        let resolved = self.resolve_type_alias(type_name);

        // Check if it's in our known Copy types
        if self.copy_types.contains(&resolved) {
            return true;
        }

        // References and raw pointers are always Copy
        if resolved.starts_with('&') || resolved.starts_with('*') {
            return true;
        }

        // Types ending in Id are usually Copy (DefId, TyId, FuncId, etc.)
        if resolved.ends_with("Id") {
            return true;
        }

        // Check for "copy_struct" marker from TypeRegistry
        if resolved == "copy_struct" {
            return true;
        }

        false
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
                    // Use Move for bindings - Copy types will be handled correctly
                    // by the Move handler (they just get copied without state change)
                    self.check_expr(init, ctx, UseKind::Move);
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
            // Call expressions - infer from function path
            ExprKind::Call { func, .. } => {
                // String::from, Vec::new, etc.
                if let ExprKind::Path(path) = &func.kind {
                    let full_path: String = path.segments.iter()
                        .map(|s| s.ident.clone())
                        .collect::<Vec<_>>()
                        .join("::");
                    // Return type inference for common constructors
                    if full_path.starts_with("String::") {
                        return Some("String".to_string());
                    }
                    if full_path.starts_with("Vec::") {
                        return Some("Vec".to_string());
                    }
                    if full_path.starts_with("Box::") {
                        return Some("Box".to_string());
                    }
                    if full_path.starts_with("HashMap::") || full_path.starts_with("HashSet::") {
                        return Some(path.segments[0].ident.clone());
                    }
                    // Return the type name if it looks like Type::method
                    if path.segments.len() >= 2 {
                        let type_name = &path.segments[0].ident;
                        // Check if first segment is capitalized (type name)
                        if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                            return Some(type_name.clone());
                        }
                    }
                }
                None
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
                    let place = Place::new(name.clone());

                    match use_kind {
                        UseKind::Read => {
                            // Check if moved
                            if ctx.is_place_moved(&place) {
                                if let Some(local) = ctx.locals.get(name) {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "use of moved value: `{}`",
                                            name
                                        ))
                                        .with_span(expr.span)
                                        .with_note(format!(
                                            "value moved here: {:?}",
                                            local.moved_span
                                        ))
                                        .with_note("move occurs because value has type that doesn't implement Copy"),
                                    );
                                }
                            }
                            // Check for mutable borrow conflict
                            if let Some(borrow) = ctx.is_mutably_borrowed(&place) {
                                self.diagnostics.write().emit(
                                    Diagnostic::error(format!(
                                        "cannot use `{}` because it was mutably borrowed",
                                        name
                                    ))
                                    .with_span(expr.span)
                                    .with_note(format!("mutable borrow occurs here: {:?}", borrow.span)),
                                );
                            }
                        }
                        UseKind::Write => {
                            if let Some(local) = ctx.locals.get(name) {
                                if !local.mutable {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "cannot assign to immutable variable `{}`",
                                            name
                                        ))
                                        .with_span(expr.span)
                                        .with_note("consider making this binding mutable: `mut`"),
                                    );
                                }
                            }
                            // Check for any active borrow (shared or mutable)
                            if let Some(borrow) = ctx.is_borrowed(&place) {
                                self.diagnostics.write().emit(
                                    Diagnostic::error(format!(
                                        "cannot assign to `{}` because it is borrowed",
                                        name
                                    ))
                                    .with_span(expr.span)
                                    .with_note(format!("borrow occurs here: {:?}", borrow.span)),
                                );
                            }
                        }
                        UseKind::Move => {
                            // Check if already moved
                            if ctx.is_place_moved(&place) {
                                self.diagnostics.write().emit(
                                    Diagnostic::error(format!(
                                        "use of moved value: `{}`",
                                        name
                                    ))
                                    .with_span(expr.span)
                                    .with_note("value used here after move"),
                                );
                            } else if let Some(local) = ctx.locals.get(name) {
                                // Check if type is Copy
                                let is_copy = local.type_name.as_ref()
                                    .map(|t| self.is_copy_type(t))
                                    .unwrap_or(false)
                                    // Fallback: use naming convention heuristics when no type info
                                    || is_likely_copy_var_name(name);

                                if !is_copy {
                                    // Check for active borrows before moving
                                    if let Some(borrow) = ctx.is_borrowed(&place) {
                                        self.diagnostics.write().emit(
                                            Diagnostic::error(format!(
                                                "cannot move out of `{}` because it is borrowed",
                                                name
                                            ))
                                            .with_span(expr.span)
                                            .with_note(format!("borrow occurs here: {:?}", borrow.span)),
                                        );
                                    } else {
                                        // Mark as moved
                                        ctx.mark_moved(place, expr.span);
                                    }
                                }
                                // Copy types just get copied, no state change needed
                            }
                        }
                        UseKind::Borrow => {
                            // Check if moved
                            if ctx.is_place_moved(&place) {
                                self.diagnostics.write().emit(
                                    Diagnostic::error(format!(
                                        "cannot borrow `{}` as it was moved",
                                        name
                                    ))
                                    .with_span(expr.span),
                                );
                            } else {
                                // Check for mutable borrow conflict
                                if let Some(borrow) = ctx.is_mutably_borrowed(&place) {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "cannot borrow `{}` as immutable because it is also borrowed as mutable",
                                            name
                                        ))
                                        .with_span(expr.span)
                                        .with_note(format!("mutable borrow occurs here: {:?}", borrow.span)),
                                    );
                                } else {
                                    // Add shared borrow
                                    ctx.add_borrow(place, BorrowKind::Shared, expr.span);
                                }
                            }
                        }
                        UseKind::MutBorrow => {
                            // Check if moved
                            if ctx.is_place_moved(&place) {
                                self.diagnostics.write().emit(
                                    Diagnostic::error(format!(
                                        "cannot borrow `{}` as mutable because it was moved",
                                        name
                                    ))
                                    .with_span(expr.span),
                                );
                            } else if let Some(local) = ctx.locals.get(name) {
                                if !local.mutable {
                                    // Skip mutability check for raw pointer types
                                    let is_raw_ptr = local.type_name.as_ref()
                                        .map(|t| t.starts_with('*'))
                                        .unwrap_or(false);
                                    if !is_raw_ptr {
                                        self.diagnostics.write().emit(
                                            Diagnostic::error(format!(
                                                "cannot borrow `{}` as mutable, as it is not declared as mutable",
                                                name
                                            ))
                                            .with_span(expr.span)
                                            .with_note("consider changing this to be mutable: `mut`"),
                                        );
                                    }
                                }
                                // Check for any existing borrow conflict
                                if let Some(borrow) = ctx.is_borrowed(&place) {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "cannot borrow `{}` as mutable because it is already borrowed",
                                            name
                                        ))
                                        .with_span(expr.span)
                                        .with_note(format!("previous borrow occurs here: {:?}", borrow.span)),
                                    );
                                } else {
                                    // Add mutable borrow
                                    ctx.add_borrow(place, BorrowKind::Mutable, expr.span);
                                }
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
            ExprKind::Struct { fields, rest, .. } => {
                // Struct initialization: check each field expression
                // fields is Vec<(String, Expr)>
                for (_field_name, field_expr) in fields {
                    // Field values are moved into the struct (or copied if Copy)
                    self.check_expr(field_expr, ctx, UseKind::Move);
                }
                // Rest expression (..rest) is also read
                if let Some(rest_expr) = rest {
                    self.check_expr(rest_expr, ctx, UseKind::Read);
                }
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
    /// Active borrows with their scope information
    active_borrows: Vec<Borrow>,
    /// Current scope ID (incremented on each new scope)
    current_scope: usize,
    /// Moved places (for partial move tracking)
    moved_places: HashSet<Place>,
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
        // Ledger types (derive Copy)
        "EventType" | "ValueState" | "Location" |
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
    // Sequence numbers (like block height in ledger)
    var_name == "seq" || var_name.ends_with("_seq") ||
    var_name == "next_seq" || var_name == "prev_seq" ||
    // Common single-letter vars for numbers
    var_name == "i" || var_name == "j" || var_name == "n" || var_name == "len" ||
    var_name == "count" ||  // Generic count variable
    // Common Copy values
    var_name == "zero" || var_name == "one" ||
    var_name == "start" || var_name == "end" ||  // Range bounds
    var_name.ends_with("_size") || var_name.ends_with("_len") ||
    var_name == "offset" || var_name.ends_with("_offset") || var_name.ends_with("_addr") ||
    var_name.ends_with("_after") || var_name.ends_with("_before") ||
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
    var_name.contains("mutable") ||
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
    var_name.ends_with("_template") || var_name == "template" ||
    // Line/column numbers (always Copy integers)
    var_name == "line" || var_name == "column" || var_name == "row" ||
    var_name.ends_with("_line") || var_name.ends_with("_column") ||
    // State/phase enums (often Copy)
    var_name == "state" || var_name.ends_with("_state") ||
    var_name == "phase" || var_name.ends_with("_phase") ||
    var_name == "status" || var_name.ends_with("_status") ||
    // Event types (usually Copy enums)
    var_name.ends_with("_event") || var_name == "event_type" ||
    // Package/metadata references (often borrowed)
    var_name == "root_package" || var_name.ends_with("_package") ||
    var_name == "package" || var_name == "metadata" ||
    // Expression and pattern variables (commonly borrowed in compilers)
    var_name == "expr" || var_name == "pat" || var_name == "stmt" ||
    var_name.ends_with("_expr") || var_name.ends_with("_pat") ||
    // Short string variables (often &str which is Copy)
    var_name == "s" || var_name == "ch" || var_name == "c"
}

impl BorrowContext {
    fn new(function_name: &str) -> Self {
        Self {
            function_name: function_name.to_string(),
            locals: IndexMap::new(),
            scope_stack: vec![Vec::new()],
            active_borrows: Vec::new(),
            current_scope: 0,
            moved_places: HashSet::new(),
        }
    }

    fn enter_scope(&mut self) -> usize {
        self.scope_stack.push(Vec::new());
        self.current_scope += 1;
        self.current_scope
    }

    fn exit_scope(&mut self, scope_id: usize) {
        // Remove locals introduced in this scope
        if let Some(names) = self.scope_stack.pop() {
            for name in names {
                self.locals.shift_remove(&name);
            }
        }
        // End borrows that were created in this scope
        self.active_borrows.retain(|b| b.scope_id < scope_id);
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

    /// Add a borrow of a place
    fn add_borrow(&mut self, place: Place, kind: BorrowKind, span: Span) {
        self.active_borrows.push(Borrow {
            place,
            kind,
            span,
            scope_id: self.current_scope,
        });
    }

    /// Check if a place is currently borrowed (shared or mutable)
    fn is_borrowed(&self, place: &Place) -> Option<&Borrow> {
        self.active_borrows.iter().find(|b| {
            // A borrow of `a` conflicts with use of `a`, `a.x`, etc.
            // A borrow of `a.x` conflicts with use of `a` or `a.x`
            b.place.is_prefix_of(place) || place.is_prefix_of(&b.place)
        })
    }

    /// Check if a place is mutably borrowed
    fn is_mutably_borrowed(&self, place: &Place) -> Option<&Borrow> {
        self.active_borrows.iter().find(|b| {
            b.kind == BorrowKind::Mutable &&
            (b.place.is_prefix_of(place) || place.is_prefix_of(&b.place))
        })
    }

    /// Check if a place or any of its parents have been moved
    fn is_place_moved(&self, place: &Place) -> bool {
        // Check if exact place is moved
        if self.moved_places.contains(place) {
            return true;
        }
        // Check if any parent is moved (e.g., if `a` is moved, `a.x` is also moved)
        for moved in &self.moved_places {
            if moved.is_prefix_of(place) {
                return true;
            }
        }
        false
    }

    /// Mark a place as moved
    fn mark_moved(&mut self, place: Place, span: Span) {
        // Update the local info for the base variable
        if let Some(info) = self.locals.get_mut(&place.base) {
            if place.projections.is_empty() {
                info.state = PlaceState::Moved;
            } else {
                info.state = PlaceState::PartiallyMoved;
            }
            info.moved_span = Some(span);
        }
        self.moved_places.insert(place);
    }
}
