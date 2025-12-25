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
}

impl BorrowChecker {
    pub fn new() -> Self {
        Self {
            diagnostics: RwLock::new(DiagnosticEmitter::new()),
        }
    }

    pub fn check_crate(&self, krate: &Crate) {
        for (_, item) in &krate.items {
            if let ItemKind::Function(f) = &item.kind {
                if let Some(ref body) = f.body {
                    let mut ctx = BorrowContext::new(&item.name);
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
                                let is_copy = local.type_name.as_ref()
                                    .map(|t| is_copy_type(t))
                                    .unwrap_or(false);

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
                                } else if local.state == PlaceState::MutablyBorrowed {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "Cannot borrow `{}` as immutable because it is already borrowed as mutable",
                                            name
                                        ))
                                        .with_span(expr.span),
                                    );
                                } else {
                                    local.state = PlaceState::Borrowed;
                                }
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
                                } else if local.state == PlaceState::Borrowed
                                    || local.state == PlaceState::MutablyBorrowed
                                {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "Cannot borrow `{}` as mutable because it is already borrowed",
                                            name
                                        ))
                                        .with_span(expr.span),
                                    );
                                } else if !local.mutable {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "Cannot borrow `{}` as mutable, as it is not declared as mutable",
                                            name
                                        ))
                                        .with_span(expr.span),
                                    );
                                } else {
                                    local.state = PlaceState::MutablyBorrowed;
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
            ExprKind::MethodCall { receiver, args, .. } => {
                self.check_expr(receiver, ctx, UseKind::Read);
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
                self.check_block(then_branch, ctx);
                if let Some(else_expr) = else_branch {
                    self.check_expr(else_expr, ctx, UseKind::Read);
                }
            }
            ExprKind::Match { expr: scrutinee, arms } => {
                self.check_expr(scrutinee, ctx, UseKind::Read);
                for arm in arms {
                    self.bind_pattern(&arm.pattern, ctx, None);
                    if let Some(ref guard) = arm.guard {
                        self.check_expr(guard, ctx, UseKind::Read);
                    }
                    self.check_expr(&arm.body, ctx, UseKind::Read);
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

struct BorrowContext {
    function_name: String,
    locals: IndexMap<String, LocalInfo>,
    scope_stack: Vec<Vec<String>>,
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
        "f32" | "f64" | "bool" | "char" | "()"
    ) ||
    // References are Copy
    type_name.starts_with("&") ||
    // Raw pointers are Copy
    type_name.starts_with("*")
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
