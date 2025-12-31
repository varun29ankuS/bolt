//! Pre-defined stubs for common external crate traits
//!
//! These allow `bolt check` to type-check code using common traits
//! without parsing the entire external crate.
//!
//! Inspired by TypeScript's .d.ts declaration files.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

static STUB_DEF_ID: AtomicU32 = AtomicU32::new(0x8000_0000); // High bit set to avoid collision

fn next_stub_id() -> u32 {
    STUB_DEF_ID.fetch_add(1, Ordering::SeqCst)
}

/// A stub trait definition
#[derive(Debug, Clone)]
pub struct TraitStub {
    pub name: String,
    pub crate_name: String,
    pub methods: Vec<MethodStub>,
    pub supertraits: Vec<String>,
    pub generics: Vec<String>, // Type parameter names
}

#[derive(Debug, Clone)]
pub struct MethodStub {
    pub name: String,
    pub takes_self: bool,
    pub self_mutable: bool,
    pub params: Vec<(String, String)>, // (name, type_name)
    pub return_type: Option<String>,
}

/// A stub type definition
#[derive(Debug, Clone)]
pub struct TypeStub {
    pub name: String,
    pub crate_name: String,
    pub kind: TypeStubKind,
    pub generics: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum TypeStubKind {
    Struct { fields: Vec<(String, String)> },
    Enum { variants: Vec<String> },
    Opaque, // We don't know the internals
}

/// Registry of all stub definitions
#[derive(Debug, Default)]
pub struct StubRegistry {
    traits: HashMap<(String, String), TraitStub>,  // (crate, name) -> stub
    types: HashMap<(String, String), TypeStub>,
    // Quick lookup: trait name -> (crate, full stub)
    trait_by_name: HashMap<String, Vec<(String, TraitStub)>>,
}

impl StubRegistry {
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_std_stubs();
        registry.register_serde_stubs();
        registry.register_common_stubs();
        registry
    }

    /// Register standard library traits
    fn register_std_stubs(&mut self) {
        // Clone trait
        self.register_trait(TraitStub {
            name: "Clone".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "clone".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![],
                    return_type: Some("Self".into()),
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // Copy trait (marker, no methods)
        self.register_trait(TraitStub {
            name: "Copy".into(),
            crate_name: "std".into(),
            methods: vec![],
            supertraits: vec!["Clone".into()],
            generics: vec![],
        });

        // Debug trait
        self.register_trait(TraitStub {
            name: "Debug".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "fmt".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![("f".into(), "&mut Formatter".into())],
                    return_type: Some("Result<(), Error>".into()),
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // Display trait
        self.register_trait(TraitStub {
            name: "Display".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "fmt".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![("f".into(), "&mut Formatter".into())],
                    return_type: Some("Result<(), Error>".into()),
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // Default trait
        self.register_trait(TraitStub {
            name: "Default".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "default".into(),
                    takes_self: false,
                    self_mutable: false,
                    params: vec![],
                    return_type: Some("Self".into()),
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // PartialEq trait
        self.register_trait(TraitStub {
            name: "PartialEq".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "eq".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![("other".into(), "&Self".into())],
                    return_type: Some("bool".into()),
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // Eq trait (marker)
        self.register_trait(TraitStub {
            name: "Eq".into(),
            crate_name: "std".into(),
            methods: vec![],
            supertraits: vec!["PartialEq".into()],
            generics: vec![],
        });

        // PartialOrd trait
        self.register_trait(TraitStub {
            name: "PartialOrd".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "partial_cmp".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![("other".into(), "&Self".into())],
                    return_type: Some("Option<Ordering>".into()),
                },
            ],
            supertraits: vec!["PartialEq".into()],
            generics: vec![],
        });

        // Ord trait
        self.register_trait(TraitStub {
            name: "Ord".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "cmp".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![("other".into(), "&Self".into())],
                    return_type: Some("Ordering".into()),
                },
            ],
            supertraits: vec!["Eq".into(), "PartialOrd".into()],
            generics: vec![],
        });

        // Hash trait
        self.register_trait(TraitStub {
            name: "Hash".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "hash".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![("state".into(), "&mut H".into())],
                    return_type: None,
                },
            ],
            supertraits: vec![],
            generics: vec!["H".into()],
        });

        // Iterator trait
        self.register_trait(TraitStub {
            name: "Iterator".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "next".into(),
                    takes_self: true,
                    self_mutable: true,
                    params: vec![],
                    return_type: Some("Option<Self::Item>".into()),
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // From trait
        self.register_trait(TraitStub {
            name: "From".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "from".into(),
                    takes_self: false,
                    self_mutable: false,
                    params: vec![("value".into(), "T".into())],
                    return_type: Some("Self".into()),
                },
            ],
            supertraits: vec![],
            generics: vec!["T".into()],
        });

        // Into trait
        self.register_trait(TraitStub {
            name: "Into".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "into".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![],
                    return_type: Some("T".into()),
                },
            ],
            supertraits: vec![],
            generics: vec!["T".into()],
        });

        // AsRef trait
        self.register_trait(TraitStub {
            name: "AsRef".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "as_ref".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![],
                    return_type: Some("&T".into()),
                },
            ],
            supertraits: vec![],
            generics: vec!["T".into()],
        });

        // AsMut trait
        self.register_trait(TraitStub {
            name: "AsMut".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "as_mut".into(),
                    takes_self: true,
                    self_mutable: true,
                    params: vec![],
                    return_type: Some("&mut T".into()),
                },
            ],
            supertraits: vec![],
            generics: vec!["T".into()],
        });

        // Drop trait
        self.register_trait(TraitStub {
            name: "Drop".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "drop".into(),
                    takes_self: true,
                    self_mutable: true,
                    params: vec![],
                    return_type: None,
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // Deref trait
        self.register_trait(TraitStub {
            name: "Deref".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "deref".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![],
                    return_type: Some("&Self::Target".into()),
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // DerefMut trait
        self.register_trait(TraitStub {
            name: "DerefMut".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "deref_mut".into(),
                    takes_self: true,
                    self_mutable: true,
                    params: vec![],
                    return_type: Some("&mut Self::Target".into()),
                },
            ],
            supertraits: vec!["Deref".into()],
            generics: vec![],
        });

        // Send trait (marker)
        self.register_trait(TraitStub {
            name: "Send".into(),
            crate_name: "std".into(),
            methods: vec![],
            supertraits: vec![],
            generics: vec![],
        });

        // Sync trait (marker)
        self.register_trait(TraitStub {
            name: "Sync".into(),
            crate_name: "std".into(),
            methods: vec![],
            supertraits: vec![],
            generics: vec![],
        });

        // Sized trait (marker)
        self.register_trait(TraitStub {
            name: "Sized".into(),
            crate_name: "std".into(),
            methods: vec![],
            supertraits: vec![],
            generics: vec![],
        });
    }

    /// Register serde traits
    fn register_serde_stubs(&mut self) {
        // Serialize trait
        self.register_trait(TraitStub {
            name: "Serialize".into(),
            crate_name: "serde".into(),
            methods: vec![
                MethodStub {
                    name: "serialize".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![("serializer".into(), "S".into())],
                    return_type: Some("Result<S::Ok, S::Error>".into()),
                },
            ],
            supertraits: vec![],
            generics: vec!["S".into()],
        });

        // Deserialize trait
        self.register_trait(TraitStub {
            name: "Deserialize".into(),
            crate_name: "serde".into(),
            methods: vec![
                MethodStub {
                    name: "deserialize".into(),
                    takes_self: false,
                    self_mutable: false,
                    params: vec![("deserializer".into(), "D".into())],
                    return_type: Some("Result<Self, D::Error>".into()),
                },
            ],
            supertraits: vec![],
            generics: vec!["'de".into(), "D".into()],
        });
    }

    /// Register other common crate traits
    fn register_common_stubs(&mut self) {
        // thiserror::Error
        self.register_trait(TraitStub {
            name: "Error".into(),
            crate_name: "thiserror".into(),
            methods: vec![],
            supertraits: vec!["Debug".into(), "Display".into()],
            generics: vec![],
        });

        // anyhow::Context
        self.register_trait(TraitStub {
            name: "Context".into(),
            crate_name: "anyhow".into(),
            methods: vec![
                MethodStub {
                    name: "context".into(),
                    takes_self: true,
                    self_mutable: false,
                    params: vec![("context".into(), "C".into())],
                    return_type: Some("Result<T, anyhow::Error>".into()),
                },
            ],
            supertraits: vec![],
            generics: vec!["T".into(), "C".into()],
        });

        // clap::Parser
        self.register_trait(TraitStub {
            name: "Parser".into(),
            crate_name: "clap".into(),
            methods: vec![
                MethodStub {
                    name: "parse".into(),
                    takes_self: false,
                    self_mutable: false,
                    params: vec![],
                    return_type: Some("Self".into()),
                },
                MethodStub {
                    name: "try_parse".into(),
                    takes_self: false,
                    self_mutable: false,
                    params: vec![],
                    return_type: Some("Result<Self, Error>".into()),
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // tokio/async traits
        self.register_trait(TraitStub {
            name: "Future".into(),
            crate_name: "std".into(),
            methods: vec![
                MethodStub {
                    name: "poll".into(),
                    takes_self: true,
                    self_mutable: false, // Pin<&mut Self>
                    params: vec![("cx".into(), "&mut Context".into())],
                    return_type: Some("Poll<Self::Output>".into()),
                },
            ],
            supertraits: vec![],
            generics: vec![],
        });

        // rayon::ParallelIterator
        self.register_trait(TraitStub {
            name: "ParallelIterator".into(),
            crate_name: "rayon".into(),
            methods: vec![],
            supertraits: vec![],
            generics: vec![],
        });

        // parking_lot types are mostly used as concrete types, not traits
        // But we stub RwLock, Mutex behavior through their methods

        // indexmap - mostly concrete types
        self.register_type(TypeStub {
            name: "IndexMap".into(),
            crate_name: "indexmap".into(),
            kind: TypeStubKind::Opaque,
            generics: vec!["K".into(), "V".into()],
        });

        self.register_type(TypeStub {
            name: "IndexSet".into(),
            crate_name: "indexmap".into(),
            kind: TypeStubKind::Opaque,
            generics: vec!["T".into()],
        });
    }

    fn register_trait(&mut self, stub: TraitStub) {
        let key = (stub.crate_name.clone(), stub.name.clone());
        self.trait_by_name
            .entry(stub.name.clone())
            .or_default()
            .push((stub.crate_name.clone(), stub.clone()));
        self.traits.insert(key, stub);
    }

    fn register_type(&mut self, stub: TypeStub) {
        let key = (stub.crate_name.clone(), stub.name.clone());
        self.types.insert(key, stub);
    }

    /// Look up a trait by crate and name
    pub fn get_trait(&self, crate_name: &str, trait_name: &str) -> Option<&TraitStub> {
        self.traits.get(&(crate_name.to_string(), trait_name.to_string()))
    }

    /// Look up a trait by name only (returns all matches from different crates)
    pub fn find_trait(&self, trait_name: &str) -> Option<&[(String, TraitStub)]> {
        self.trait_by_name.get(trait_name).map(|v| v.as_slice())
    }

    /// Look up a type by crate and name
    pub fn get_type(&self, crate_name: &str, type_name: &str) -> Option<&TypeStub> {
        self.types.get(&(crate_name.to_string(), type_name.to_string()))
    }

    /// Check if a trait exists (in any crate)
    pub fn has_trait(&self, trait_name: &str) -> bool {
        self.trait_by_name.contains_key(trait_name)
    }

    /// Get all registered trait names
    pub fn all_trait_names(&self) -> impl Iterator<Item = &str> {
        self.trait_by_name.keys().map(|s| s.as_str())
    }
}

/// Global stub registry (lazy initialized)
pub fn global_stubs() -> &'static StubRegistry {
    use once_cell::sync::Lazy;
    static STUBS: Lazy<StubRegistry> = Lazy::new(StubRegistry::new);
    &STUBS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_registry() {
        let stubs = global_stubs();

        // Check std traits
        assert!(stubs.has_trait("Clone"));
        assert!(stubs.has_trait("Debug"));
        assert!(stubs.has_trait("Default"));

        // Check serde traits
        assert!(stubs.has_trait("Serialize"));
        assert!(stubs.has_trait("Deserialize"));

        // Check specific lookup
        let clone = stubs.get_trait("std", "Clone").unwrap();
        assert_eq!(clone.methods.len(), 1);
        assert_eq!(clone.methods[0].name, "clone");
    }
}
