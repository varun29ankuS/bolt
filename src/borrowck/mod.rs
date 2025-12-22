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
            StmtKind::Let { pattern, init, .. } => {
                if let Some(init) = init {
                    self.check_expr(init, ctx, UseKind::Read);
                }
                self.bind_pattern(pattern, ctx);
            }
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
                self.check_expr(expr, ctx, UseKind::Read);
            }
            StmtKind::Item(_) => {}
        }
    }

    fn bind_pattern(&self, pattern: &Pattern, ctx: &mut BorrowContext) {
        match &pattern.kind {
            PatternKind::Ident { name, mutable, .. } => {
                ctx.introduce_local(name.clone(), *mutable, pattern.span);
            }
            PatternKind::Tuple(pats) => {
                for pat in pats {
                    self.bind_pattern(pat, ctx);
                }
            }
            PatternKind::Struct { fields, .. } => {
                for field in fields {
                    self.bind_pattern(&field.pattern, ctx);
                }
            }
            PatternKind::TupleStruct { elems, .. } => {
                for elem in elems {
                    self.bind_pattern(elem, ctx);
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
                                if local.state == PlaceState::Moved {
                                    self.diagnostics.write().emit(
                                        Diagnostic::error(format!(
                                            "Use of moved value: `{}`",
                                            name
                                        ))
                                        .with_span(expr.span),
                                    );
                                } else {
                                    local.state = PlaceState::Moved;
                                    local.moved_span = Some(expr.span);
                                }
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
                    self.bind_pattern(&arm.pattern, ctx);
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
                self.bind_pattern(pattern, ctx);
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
                self.check_expr(base, ctx, use_kind);
            }
            ExprKind::Index { expr: base, index } => {
                self.check_expr(base, ctx, use_kind);
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

    fn introduce_local(&mut self, name: String, mutable: bool, span: Span) {
        self.locals.insert(
            name.clone(),
            LocalInfo {
                mutable,
                state: PlaceState::Valid,
                defined_span: span,
                moved_span: None,
            },
        );
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.push(name);
        }
    }
}
