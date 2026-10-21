//! What this machine is and how busy it is, for `larmor bench`.

use std::process::Command;

use serde::Serialize;

/// Hardware and OS, as far as the system tells.
#[derive(Clone, Debug, Serialize)]
pub struct Machine {
    pub chip: String,
    pub cpu_cores: usize,
    /// Performance and efficiency cores on Apple silicon.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub performance_cores: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub efficiency_cores: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_cores: Option<usize>,
    pub memory_gb: f64,
    pub os: String,
    pub arch: String,
}

/// How busy the machine was when a timing started.
#[derive(Clone, Debug, Serialize)]
pub struct Load {
    /// One-minute load average.
    pub load_average: Option<f64>,
    /// GPU utilization in percent (macOS `IOAccelerator` statistics).
    pub gpu_utilization: Option<u32>,
}

impl Load {
    pub fn now() -> Load {
        let load_average = if cfg!(target_os = "macos") {
            sysctl("vm.loadavg")
                .and_then(|s| s.trim_matches(['{', '}', ' ']).split_whitespace().next()?.parse().ok())
        } else {
            std::fs::read_to_string("/proc/loadavg")
                .ok()
                .and_then(|s| s.split_whitespace().next()?.parse().ok())
        };
        let gpu_utilization =
            ioreg_accelerator().and_then(|text| ioreg_number(&text, "\"Device Utilization %\"="));
        Load { load_average, gpu_utilization: gpu_utilization.map(|v| v as u32) }
    }

    pub fn describe(&self) -> String {
        let load = self.load_average.map_or("?".into(), |l| format!("{l:.2}"));
        let gpu = self.gpu_utilization.map_or("?".into(), |g| format!("{g}%"));
        format!("load average {load}, GPU busy {gpu}")
    }
}

impl Machine {
    pub fn detect() -> Machine {
        let cpu_cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let arch = std::env::consts::ARCH.to_string();
        if cfg!(target_os = "macos") {
            let accelerator = ioreg_accelerator();
            Machine {
                chip: sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into()),
                cpu_cores: sysctl("hw.ncpu").and_then(|s| s.parse().ok()).unwrap_or(cpu_cores),
                performance_cores: sysctl("hw.perflevel0.physicalcpu").and_then(|s| s.parse().ok()),
                efficiency_cores: sysctl("hw.perflevel1.physicalcpu").and_then(|s| s.parse().ok()),
                gpu_cores: accelerator.and_then(|text| ioreg_number(&text, "\"gpu-core-count\" = ")),
                memory_gb: sysctl("hw.memsize")
                    .and_then(|s| s.parse::<f64>().ok())
                    .map_or(0.0, |b| b / (1u64 << 30) as f64),
                os: command("sw_vers", &["-productVersion"]).map_or("macOS".into(), |v| format!("macOS {v}")),
                arch,
            }
        } else {
            let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
            let chip = cpuinfo
                .lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map_or("unknown".into(), |s| s.trim().to_string());
            let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
            let memory_kb: f64 = meminfo
                .lines()
                .find(|l| l.starts_with("MemTotal"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
                .unwrap_or(0.0);
            let os_release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
            let os = os_release
                .lines()
                .find_map(|l| l.strip_prefix("PRETTY_NAME="))
                .map_or(std::env::consts::OS.to_string(), |s| s.trim_matches('"').to_string());
            Machine {
                chip,
                cpu_cores,
                performance_cores: None,
                efficiency_cores: None,
                gpu_cores: None,
                memory_gb: memory_kb / (1u64 << 20) as f64,
                os,
                arch,
            }
        }
    }

    pub fn describe(&self) -> String {
        let cores = match (self.performance_cores, self.efficiency_cores) {
            (Some(p), Some(e)) => format!("{} CPU cores ({p}P + {e}E)", self.cpu_cores),
            _ => format!("{} CPU cores", self.cpu_cores),
        };
        let gpu = self.gpu_cores.map_or(String::new(), |g| format!(", {g} GPU cores"));
        format!("{}, {cores}{gpu}, {:.0} GB, {} ({})", self.chip, self.memory_gb, self.os, self.arch)
    }
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn sysctl(name: &str) -> Option<String> {
    command("sysctl", &["-n", name])
}

/// `ioreg` output for the GPU, which carries its core count and utilization (macOS only).
fn ioreg_accelerator() -> Option<String> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    command("ioreg", &["-r", "-d", "1", "-c", "IOAccelerator"])
}

/// The integer after `key` in `text`.
fn ioreg_number(text: &str, key: &str) -> Option<usize> {
    let start = text.find(key)? + key.len();
    let digits: String = text[start..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_ioreg_numbers() {
        let text = r#"| "gpu-core-count" = 60 ... "Device Utilization %"=7,"#;
        assert_eq!(ioreg_number(text, "\"gpu-core-count\" = "), Some(60));
        assert_eq!(ioreg_number(text, "\"Device Utilization %\"="), Some(7));
        assert_eq!(ioreg_number(text, "\"missing\"="), None);
    }
}
