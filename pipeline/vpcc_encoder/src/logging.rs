use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogMode {
    ConsoleOnly,
    FileOnly,
    Both,
}

impl LogMode {
    pub fn from_env() -> Self {
        match std::env::var("VPCC_ENCODER_LOG_MODE")
            .ok()
            .as_deref()
            .unwrap_or("both")
        {
            "console" => Self::ConsoleOnly,
            "file" => Self::FileOnly,
            _ => Self::Both,
        }
    }
}

pub struct RunLogger {
    mode: LogMode,
    file: Option<BufWriter<File>>,
    path: Option<PathBuf>,
}

impl RunLogger {
    pub fn new(
        output_dir: &Path,
        override_path: Option<&Path>,
        mode: LogMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = override_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| output_dir.join("run.log"));

        let file = match mode {
            LogMode::ConsoleOnly => None,
            LogMode::FileOnly | LogMode::Both => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                Some(BufWriter::new(File::create(&path)?))
            }
        };

        let path = if file.is_some() { Some(path) } else { None };

        Ok(Self { mode, file, path })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn summary_line(&mut self, line: impl AsRef<str>) -> std::io::Result<()> {
        let line = line.as_ref();
        if matches!(self.mode, LogMode::ConsoleOnly | LogMode::Both) {
            println!("{line}");
        }
        if let Some(file) = self.file.as_mut() {
            writeln!(file, "{line}")?;
        }
        Ok(())
    }

    pub fn summary_lines(
        &mut self,
        lines: impl IntoIterator<Item = String>,
    ) -> std::io::Result<()> {
        for line in lines {
            self.summary_line(line)?;
        }
        Ok(())
    }

    pub fn file_lines(&mut self, lines: impl IntoIterator<Item = String>) -> std::io::Result<()> {
        if let Some(file) = self.file.as_mut() {
            for line in lines {
                writeln!(file, "{line}")?;
            }
        }
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(file) = self.file.as_mut() {
            file.flush()?;
        }
        Ok(())
    }
}
