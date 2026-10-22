//! Writing files so that neither a reader nor an interrupted run ever sees half of one.

use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Writes `bytes` to `path` atomically; see [`write_atomic_with`].
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    write_atomic_with(path, |file| file.write_all(bytes))
}

/// Writes `path` atomically: `write` fills a uniquely named temporary file in the same folder
/// (created if needed), which is synced to disk and then renamed over `path`.
///
/// If `write` fails, or the process stops before the rename, `path` keeps its old contents (or
/// stays absent) and the temporary file is removed. The temporary name is random, so concurrent
/// writers, even in different containers that share a folder and a process id, never write into
/// the same temporary file; the last rename wins with a complete file.
pub fn write_atomic_with(
    path: &Path,
    write: impl FnOnce(&mut File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(dir)?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut tmp = tempfile::Builder::new().prefix(&format!(".{name}.")).suffix(".tmp").tempfile_in(dir)?;
    write(tmp.as_file_mut())?;
    tmp.as_file().sync_all()?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn a_failed_write_leaves_the_target_and_no_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.bin");
        std::fs::write(&path, b"old contents").unwrap();
        let err = write_atomic_with(&path, |file| {
            file.write_all(b"half of the new")?;
            Err(std::io::Error::other("disk full"))
        })
        .unwrap_err();
        assert_eq!(err.to_string(), "disk full");
        assert_eq!(std::fs::read(&path).unwrap(), b"old contents");
        assert_eq!(entries(dir.path()), ["out.bin"]);

        let absent = dir.path().join("sub/new.bin");
        assert!(write_atomic_with(&absent, |_| Err(std::io::Error::other("no"))).is_err());
        assert!(!absent.exists());
    }

    #[test]
    fn concurrent_writers_leave_one_complete_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entry.bin");
        let contents: Vec<Vec<u8>> = (0..8u8).map(|i| vec![i; 100_000 + 1000 * usize::from(i)]).collect();
        std::thread::scope(|scope| {
            for bytes in &contents {
                let path = &path;
                scope.spawn(move || write_atomic(path, bytes).unwrap());
            }
        });
        let written = std::fs::read(&path).unwrap();
        assert!(contents.contains(&written), "the file is a mix of writes");
        assert_eq!(entries(dir.path()), ["entry.bin"]);
    }
}
