//! Derive macro expansion for common traits
//!
//! This module generates impl blocks for #[derive(...)] attributes.
//! Supported: Clone, Debug, Default, Copy, PartialEq, Eq, Hash
//!
//! Note: This is called from the parser when it detects derive attributes.

use crate::error::Span;
use crate::hir::*;
use std::sync::atomic::{AtomicU32, Ordering};

static DERIVE_DEF_ID: AtomicU32 = AtomicU32::new(0xD000_0000);

fn next_derive_id() -> DefId {
    DERIVE_DEF_ID.fetch_add(1, Ordering::SeqCst)
}

/// Expand derive macros for a struct, returning generated impl items
pub fn expand_derive_for_struct(
    struct_name: &str,
    fields: &[(String, Type)],
    derives: &[String],
    span: Span,
) -> Vec<(DefId, Item)> {
    let mut impls = Vec::new();

    for derive in derives {
        if let Some(item) = expand_single(struct_name, fields, derive, span) {
            let def_id = item.id;
            impls.push((def_id, item));
        }
    }

    impls
}

fn expand_single(
    struct_name: &str,
    _fields: &[(String, Type)],
    derive: &str,
    span: Span,
) -> Option<Item> {
    match derive {
        "Clone" | "Debug" | "Default" | "Copy" | "PartialEq" | "Eq" | "Hash" => {
            Some(gen_trait_impl(struct_name, derive, span))
        }
        _ => None,
    }
}

/// Generate a trait impl (simplified - just registers the impl, no method bodies)
fn gen_trait_impl(struct_name: &str, trait_name: &str, span: Span) -> Item {
    let path = Path {
        segments: vec![PathSegment {
            ident: struct_name.to_string(),
            args: None,
        }],
    };

    let trait_path = Path {
        segments: vec![PathSegment {
            ident: trait_name.to_string(),
            args: None,
        }],
    };

    Item {
        id: next_derive_id(),
        name: format!("impl_{}_for_{}", trait_name, struct_name),
        kind: ItemKind::Impl(Impl {
            generics: Generics::default(),
            trait_ref: Some(trait_path),
            self_ty: Type {
                kind: TypeKind::Path(path),
                span,
            },
            items: vec![],
            assoc_types: vec![],
        }),
        visibility: Visibility::Private,
        span,
    }
}

/// Parse derive attributes and return list of trait names
pub fn parse_derive_attrs(attrs: &[(String, String)]) -> Vec<String> {
    let mut derives = Vec::new();

    for (name, value) in attrs {
        if name == "derive" {
            // Parse value like "Clone, Debug, Default"
            for part in value.split(',') {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    derives.push(trimmed.to_string());
                }
            }
        }
    }

    derives
}
