//! Terminal output: colors (off when stdout is not a terminal or `NO_COLOR` is set) and number
//! formatting shared by the commands.

use std::io::IsTerminal;
use std::sync::OnceLock;

fn color_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none())
}

fn paint(code: &str, s: &str) -> String {
    if color_enabled() { format!("\x1b[{code}m{s}\x1b[0m") } else { s.to_string() }
}

pub fn bold(s: &str) -> String {
    paint("1", s)
}

pub fn dim(s: &str) -> String {
    paint("2", s)
}

pub fn red(s: &str) -> String {
    paint("31", s)
}

pub fn green(s: &str) -> String {
    paint("32", s)
}

pub fn yellow(s: &str) -> String {
    paint("33", s)
}

/// `0.123 s` or `12.3 s`.
pub fn secs(seconds: f64) -> String {
    if seconds < 10.0 { format!("{seconds:.3} s") } else { format!("{seconds:.1} s") }
}

/// A metric with its 95% bootstrap interval, `0.812 [0.701, 0.893]`.
pub fn with_ci(value: f64, ci: [f64; 2]) -> String {
    format!("{value:.3} [{:.3}, {:.3}]", ci[0], ci[1])
}
