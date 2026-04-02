//! HT brand accent colors.
//!
//! Primary: light teal.  Secondary: purple.
//! These are used throughout the TUI wherever the upstream codebase used
//! `Color::Cyan` as a generic accent.

use ratatui::style::Color;

/// Primary accent — light teal.
pub const PRIMARY: Color = Color::Rgb(0x4D, 0xB6, 0xAC); // #4DB6AC

/// Secondary accent — purple (used where a contrasting accent is helpful).
pub const SECONDARY: Color = Color::Rgb(0x7E, 0x57, 0xC2); // #7E57C2
