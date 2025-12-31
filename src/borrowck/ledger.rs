//! Ownership Ledger - Blockchain-inspired borrow tracking
//!
//! Every ownership change is recorded as an immutable event.
//! This provides:
//! - Full audit trail for debugging
//! - Clear error explanations
//! - LLM-friendly structured output
//! - Time-travel debugging (what was the state at line N?)

use crate::error::Span;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single ownership event in the ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnershipEvent {
    /// Unique sequence number (like block height)
    pub seq: u64,
    /// The variable/value this event concerns
    pub value: String,
    /// Type of ownership change
    pub event_type: EventType,
    /// Source location
    pub location: Location,
    /// Reference to related event (e.g., borrow references create)
    pub references: Option<u64>,
    /// State of the value after this event
    pub state_after: ValueState,
    /// Additional context for error messages
    pub context: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Value created (let x = ...)
    Create,
    /// Value moved to another binding (let y = x)
    Move,
    /// Shared borrow created (&x)
    Borrow,
    /// Mutable borrow created (&mut x)
    BorrowMut,
    /// Borrow ended (reference goes out of scope)
    EndBorrow,
    /// Value dropped (goes out of scope)
    Drop,
    /// Value used (read)
    Use,
    /// Value mutated (write)
    Mutate,
    /// Attempt to use - may be valid or error
    Access,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueState {
    /// Value is valid and owned
    Owned,
    /// Value has been moved, no longer accessible
    Moved,
    /// Value is borrowed (shared)
    Borrowed { count: u32 },
    /// Value is mutably borrowed
    MutablyBorrowed,
    /// Value has been dropped
    Dropped,
    /// Partial move (some fields moved)
    PartiallyMoved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl Location {
    pub fn from_span(span: Span, file_name: &str) -> Self {
        // For now, use span offsets as line numbers (simplified)
        Self {
            file: file_name.to_string(),
            line: span.start as u32,
            column: 0,
        }
    }

    pub fn unknown() -> Self {
        Self {
            file: "<unknown>".to_string(),
            line: 0,
            column: 0,
        }
    }
}

/// The ownership ledger - append-only log of all ownership events
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OwnershipLedger {
    /// All events in order
    events: Vec<OwnershipEvent>,
    /// Current sequence number
    next_seq: u64,
    /// Current state of each value (computed from events)
    current_state: HashMap<String, ValueState>,
    /// Map of value -> event indices for quick lookup
    value_events: HashMap<String, Vec<usize>>,
    /// Active borrows: borrowed_value -> (borrower, event_seq)
    active_borrows: HashMap<String, Vec<(String, u64)>>,
}

impl OwnershipLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new event and return its sequence number
    pub fn record(&mut self,
        value: String,
        event_type: EventType,
        location: Location,
        references: Option<u64>,
        context: Option<String>,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        // Compute new state based on event type and current state
        let current = self.current_state.get(&value).copied().unwrap_or(ValueState::Owned);
        let state_after = self.compute_new_state(current, event_type);

        let event = OwnershipEvent {
            seq,
            value: value.clone(),
            event_type,
            location,
            references,
            state_after,
            context,
        };

        // Update indexes
        let idx = self.events.len();
        self.events.push(event);
        self.value_events.entry(value.clone()).or_default().push(idx);
        self.current_state.insert(value, state_after);

        seq
    }

    /// Record value creation
    pub fn create(&mut self, value: &str, location: Location, type_info: Option<&str>) -> u64 {
        self.record(
            value.to_string(),
            EventType::Create,
            location,
            None,
            type_info.map(|t| format!("type: {}", t)),
        )
    }

    /// Record a move
    pub fn move_value(&mut self, from: &str, to: &str, location: Location) -> u64 {
        // Find the create event for 'from'
        let create_seq = self.find_create_event(from);

        // Record the move
        let seq = self.record(
            from.to_string(),
            EventType::Move,
            location.clone(),
            create_seq,
            Some(format!("moved to `{}`", to)),
        );

        // Also create the new value
        self.record(
            to.to_string(),
            EventType::Create,
            location,
            Some(seq),
            Some(format!("moved from `{}`", from)),
        );

        seq
    }

    /// Record a borrow
    pub fn borrow(&mut self, value: &str, borrower: &str, mutable: bool, location: Location) -> u64 {
        let event_type = if mutable { EventType::BorrowMut } else { EventType::Borrow };
        let create_seq = self.find_create_event(value);

        let seq = self.record(
            value.to_string(),
            event_type,
            location,
            create_seq,
            Some(format!("borrowed by `{}`", borrower)),
        );

        // Track active borrow
        self.active_borrows
            .entry(value.to_string())
            .or_default()
            .push((borrower.to_string(), seq));

        seq
    }

    /// Record end of borrow
    pub fn end_borrow(&mut self, value: &str, borrower: &str, location: Location) -> u64 {
        // Find the borrow event
        let borrow_seq = self.active_borrows
            .get(value)
            .and_then(|borrows| borrows.iter().find(|(b, _)| b == borrower))
            .map(|(_, seq)| *seq);

        // Remove from active borrows
        if let Some(borrows) = self.active_borrows.get_mut(value) {
            borrows.retain(|(b, _)| b != borrower);
        }

        self.record(
            value.to_string(),
            EventType::EndBorrow,
            location,
            borrow_seq,
            Some(format!("borrow by `{}` ended", borrower)),
        )
    }

    /// Record a use (read access)
    pub fn use_value(&mut self, value: &str, location: Location) -> Result<u64, OwnershipError> {
        let state = self.current_state.get(value).copied().unwrap_or(ValueState::Owned);

        match state {
            ValueState::Moved => {
                let move_event = self.find_last_event(value, EventType::Move);
                Err(OwnershipError::UseAfterMove {
                    value: value.to_string(),
                    use_location: location,
                    move_event: move_event.cloned(),
                    history: self.get_value_history(value),
                })
            }
            ValueState::Dropped => {
                Err(OwnershipError::UseAfterDrop {
                    value: value.to_string(),
                    use_location: location,
                    history: self.get_value_history(value),
                })
            }
            _ => {
                Ok(self.record(
                    value.to_string(),
                    EventType::Use,
                    location,
                    None,
                    None,
                ))
            }
        }
    }

    /// Record a mutation (write access)
    pub fn mutate_value(&mut self, value: &str, location: Location) -> Result<u64, OwnershipError> {
        let state = self.current_state.get(value).copied().unwrap_or(ValueState::Owned);

        match state {
            ValueState::Moved => {
                let move_event = self.find_last_event(value, EventType::Move);
                Err(OwnershipError::UseAfterMove {
                    value: value.to_string(),
                    use_location: location,
                    move_event: move_event.cloned(),
                    history: self.get_value_history(value),
                })
            }
            ValueState::Borrowed { .. } => {
                Err(OwnershipError::MutateWhileBorrowed {
                    value: value.to_string(),
                    mutate_location: location,
                    active_borrows: self.get_active_borrows(value),
                    history: self.get_value_history(value),
                })
            }
            _ => {
                Ok(self.record(
                    value.to_string(),
                    EventType::Mutate,
                    location,
                    None,
                    None,
                ))
            }
        }
    }

    /// Check if a borrow is valid
    pub fn check_borrow(&self, value: &str, mutable: bool, location: &Location) -> Result<(), OwnershipError> {
        let state = self.current_state.get(value).copied().unwrap_or(ValueState::Owned);

        match (state, mutable) {
            (ValueState::Moved, _) => {
                Err(OwnershipError::BorrowOfMoved {
                    value: value.to_string(),
                    borrow_location: location.clone(),
                    history: self.get_value_history(value),
                })
            }
            (ValueState::MutablyBorrowed, _) => {
                Err(OwnershipError::AlreadyMutablyBorrowed {
                    value: value.to_string(),
                    new_borrow_location: location.clone(),
                    existing_borrow: self.find_last_event(value, EventType::BorrowMut).cloned(),
                    history: self.get_value_history(value),
                })
            }
            (ValueState::Borrowed { .. }, true) => {
                Err(OwnershipError::MutBorrowWhileBorrowed {
                    value: value.to_string(),
                    mut_borrow_location: location.clone(),
                    active_borrows: self.get_active_borrows(value),
                    history: self.get_value_history(value),
                })
            }
            _ => Ok(())
        }
    }

    // Helper methods

    fn compute_new_state(&self, current: ValueState, event: EventType) -> ValueState {
        match event {
            EventType::Create => ValueState::Owned,
            EventType::Move => ValueState::Moved,
            EventType::Drop => ValueState::Dropped,
            EventType::Borrow => {
                match current {
                    ValueState::Borrowed { count } => ValueState::Borrowed { count: count + 1 },
                    _ => ValueState::Borrowed { count: 1 },
                }
            }
            EventType::BorrowMut => ValueState::MutablyBorrowed,
            EventType::EndBorrow => {
                match current {
                    ValueState::Borrowed { count } if count > 1 => ValueState::Borrowed { count: count - 1 },
                    ValueState::Borrowed { .. } | ValueState::MutablyBorrowed => ValueState::Owned,
                    other => other,
                }
            }
            EventType::Use | EventType::Mutate | EventType::Access => current,
        }
    }

    fn find_create_event(&self, value: &str) -> Option<u64> {
        self.value_events.get(value)
            .and_then(|indices| {
                indices.iter()
                    .find(|&&idx| self.events[idx].event_type == EventType::Create)
                    .map(|&idx| self.events[idx].seq)
            })
    }

    fn find_last_event(&self, value: &str, event_type: EventType) -> Option<&OwnershipEvent> {
        self.value_events.get(value)
            .and_then(|indices| {
                indices.iter().rev()
                    .find(|&&idx| self.events[idx].event_type == event_type)
                    .map(|&idx| &self.events[idx])
            })
    }

    /// Get the full history of a value (for error messages)
    pub fn get_value_history(&self, value: &str) -> Vec<OwnershipEvent> {
        self.value_events.get(value)
            .map(|indices| indices.iter().map(|&idx| self.events[idx].clone()).collect())
            .unwrap_or_default()
    }

    fn get_active_borrows(&self, value: &str) -> Vec<(String, u64)> {
        self.active_borrows.get(value).cloned().unwrap_or_default()
    }

    /// Get current state of a value
    pub fn get_state(&self, value: &str) -> Option<ValueState> {
        self.current_state.get(value).copied()
    }

    /// Export ledger as JSON (LLM-friendly)
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Format history as a table (human-friendly)
    pub fn format_history(&self, value: &str) -> String {
        let mut output = String::new();
        output.push_str(&format!("\nOwnership Ledger for `{}`:\n", value));
        output.push_str("┌─────┬────────────┬──────────┬─────────────────────────┐\n");
        output.push_str("│ #   │ Event      │ Location │ State After             │\n");
        output.push_str("├─────┼────────────┼──────────┼─────────────────────────┤\n");

        for event in self.get_value_history(value) {
            let event_str = format!("{:?}", event.event_type);
            let loc_str = format!("line {}", event.location.line);
            let state_str = format!("{:?}", event.state_after);
            output.push_str(&format!(
                "│ {:<3} │ {:<10} │ {:<8} │ {:<23} │\n",
                event.seq, event_str, loc_str, state_str
            ));
        }

        output.push_str("└─────┴────────────┴──────────┴─────────────────────────┘\n");
        output
    }
}

/// Ownership errors with full context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OwnershipError {
    UseAfterMove {
        value: String,
        use_location: Location,
        move_event: Option<OwnershipEvent>,
        history: Vec<OwnershipEvent>,
    },
    UseAfterDrop {
        value: String,
        use_location: Location,
        history: Vec<OwnershipEvent>,
    },
    BorrowOfMoved {
        value: String,
        borrow_location: Location,
        history: Vec<OwnershipEvent>,
    },
    AlreadyMutablyBorrowed {
        value: String,
        new_borrow_location: Location,
        existing_borrow: Option<OwnershipEvent>,
        history: Vec<OwnershipEvent>,
    },
    MutBorrowWhileBorrowed {
        value: String,
        mut_borrow_location: Location,
        active_borrows: Vec<(String, u64)>,
        history: Vec<OwnershipEvent>,
    },
    MutateWhileBorrowed {
        value: String,
        mutate_location: Location,
        active_borrows: Vec<(String, u64)>,
        history: Vec<OwnershipEvent>,
    },
}

impl OwnershipError {
    /// Format error for humans with full history
    pub fn format_human(&self) -> String {
        match self {
            OwnershipError::UseAfterMove { value, use_location, move_event, history } => {
                let mut msg = format!(
                    "error: use of moved value `{}`\n  --> {}:{}\n",
                    value, use_location.file, use_location.line
                );
                if let Some(mv) = move_event {
                    msg.push_str(&format!(
                        "  |\n  = note: value moved at line {}\n",
                        mv.location.line
                    ));
                }
                msg.push_str(&format_history_mini(history));
                msg
            }
            // ... other error types
            _ => format!("{:?}", self),
        }
    }

    /// Format error as JSON for LLMs
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

fn format_history_mini(history: &[OwnershipEvent]) -> String {
    let mut msg = String::new();
    msg.push_str("  = ownership history:\n");
    for event in history.iter().take(10) {
        msg.push_str(&format!(
            "    [{:?}] at line {} → {:?}\n",
            event.event_type, event.location.line, event.state_after
        ));
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ledger() {
        let mut ledger = OwnershipLedger::new();

        // Create a value
        ledger.create("x", Location { file: "test.rs".into(), line: 1, column: 0 }, Some("String"));

        // Use it
        assert!(ledger.use_value("x", Location { file: "test.rs".into(), line: 2, column: 0 }).is_ok());

        // Move it
        ledger.move_value("x", "y", Location { file: "test.rs".into(), line: 3, column: 0 });

        // Try to use after move - should fail
        assert!(ledger.use_value("x", Location { file: "test.rs".into(), line: 4, column: 0 }).is_err());

        // Using y should work
        assert!(ledger.use_value("y", Location { file: "test.rs".into(), line: 5, column: 0 }).is_ok());
    }

    #[test]
    fn test_borrow_tracking() {
        let mut ledger = OwnershipLedger::new();

        ledger.create("data", Location { file: "test.rs".into(), line: 1, column: 0 }, None);

        // Borrow it
        ledger.borrow("data", "ref1", false, Location { file: "test.rs".into(), line: 2, column: 0 });

        // Can still read
        assert!(ledger.use_value("data", Location { file: "test.rs".into(), line: 3, column: 0 }).is_ok());

        // Can't mutate while borrowed
        assert!(ledger.mutate_value("data", Location { file: "test.rs".into(), line: 4, column: 0 }).is_err());

        // End borrow
        ledger.end_borrow("data", "ref1", Location { file: "test.rs".into(), line: 5, column: 0 });

        // Now can mutate
        assert!(ledger.mutate_value("data", Location { file: "test.rs".into(), line: 6, column: 0 }).is_ok());
    }
}
