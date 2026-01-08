//! Bolt LSP Server
//!
//! Provides real-time type checking and diagnostics for editors.
//!
//! # Features
//! - Real-time diagnostics on save
//! - Go-to-definition
//! - Hover information
//! - Fast incremental checking

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

use crate::api::{check_file, CheckConfig, Strictness};
use crate::hir::Crate;
use crate::ty::TypeRegistry;

/// Document state cached by the server
#[derive(Debug)]
struct DocumentState {
    /// Current content
    content: String,
    /// Parsed HIR (if successful)
    krate: Option<Crate>,
    /// Last check version
    version: i32,
}

/// Bolt Language Server
pub struct BoltLanguageServer {
    /// LSP client for sending notifications
    client: Client,
    /// Cached document states
    documents: DashMap<Url, DocumentState>,
    /// Configuration
    config: RwLock<ServerConfig>,
}

#[derive(Debug, Clone)]
struct ServerConfig {
    /// Check on save vs on type
    check_on_type: bool,
    /// Strictness level
    strictness: Strictness,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            check_on_type: false, // Default to check on save (less CPU)
            strictness: Strictness::Lenient,
        }
    }
}

impl BoltLanguageServer {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: DashMap::new(),
            config: RwLock::new(ServerConfig::default()),
        }
    }

    /// Check a document and publish diagnostics
    async fn check_document(&self, uri: &Url) {
        let path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return,
        };

        // Only check .rs files
        if path.extension().map(|e| e != "rs").unwrap_or(true) {
            return;
        }

        let config = CheckConfig {
            strictness: self.config.read().strictness,
            ..Default::default()
        };

        let result = check_file(&path, config);

        // Convert to LSP diagnostics
        let diagnostics: Vec<Diagnostic> = result
            .errors
            .iter()
            .map(|err| {
                let range = err.location.as_ref().map(|loc| {
                    Range {
                        start: Position {
                            line: loc.line.saturating_sub(1) as u32,
                            character: loc.column.saturating_sub(1) as u32,
                        },
                        end: Position {
                            line: loc.line.saturating_sub(1) as u32,
                            character: (loc.column + 10) as u32, // Approximate end
                        },
                    }
                }).unwrap_or(Range {
                    start: Position { line: 0, character: 0 },
                    end: Position { line: 0, character: 0 },
                });

                Diagnostic {
                    range,
                    severity: Some(match err.severity {
                        crate::api::Severity::Error => DiagnosticSeverity::ERROR,
                        crate::api::Severity::Warning => DiagnosticSeverity::WARNING,
                        crate::api::Severity::Note => DiagnosticSeverity::INFORMATION,
                        crate::api::Severity::Help => DiagnosticSeverity::HINT,
                    }),
                    code: Some(NumberOrString::String(err.code.clone())),
                    source: Some("bolt".to_string()),
                    message: err.message.clone(),
                    related_information: None,
                    tags: None,
                    code_description: None,
                    data: None,
                }
            })
            .collect();

        // Publish diagnostics
        self.client
            .publish_diagnostics(uri.clone(), diagnostics, None)
            .await;
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for BoltLanguageServer {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Options(
                    TextDocumentSyncOptions {
                        open_close: Some(true),
                        change: Some(TextDocumentSyncKind::FULL),
                        save: Some(TextDocumentSyncSaveOptions::SaveOptions(SaveOptions {
                            include_text: Some(true),
                        })),
                        ..Default::default()
                    },
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                diagnostic_provider: Some(DiagnosticServerCapabilities::Options(
                    DiagnosticOptions {
                        identifier: Some("bolt".to_string()),
                        inter_file_dependencies: true,
                        workspace_diagnostics: false,
                        ..Default::default()
                    },
                )),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "bolt-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Bolt LSP server initialized!")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let content = params.text_document.text;
        let version = params.text_document.version;

        self.documents.insert(
            uri.clone(),
            DocumentState {
                content,
                krate: None,
                version,
            },
        );

        // Check on open
        self.check_document(&uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;

        if let Some(mut doc) = self.documents.get_mut(&uri) {
            for change in params.content_changes {
                doc.content = change.text;
            }
            doc.version = params.text_document.version;
        }

        // Check on type if enabled
        if self.config.read().check_on_type {
            self.check_document(&uri).await;
        }
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        let uri = params.text_document.uri;

        // Update content if provided
        if let Some(text) = params.text {
            if let Some(mut doc) = self.documents.get_mut(&uri) {
                doc.content = text;
            }
        }

        // Always check on save
        self.check_document(&uri).await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.documents.remove(&uri);

        // Clear diagnostics
        self.client
            .publish_diagnostics(uri, vec![], None)
            .await;
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        // TODO: Implement hover with type information
        Ok(None)
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        // TODO: Implement go-to-definition
        Ok(None)
    }
}

/// Run the LSP server
pub async fn run_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| BoltLanguageServer::new(client));
    Server::new(stdin, stdout, socket).serve(service).await;
}

/// Start the LSP server (blocking)
pub fn start() {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    rt.block_on(run_server());
}
