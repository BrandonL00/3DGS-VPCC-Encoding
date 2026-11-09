//! Optional, best-effort Discord webhook notifications for long codec runs.
//!
//! Set `VPCC_DISCORD_WEBHOOK_URL` to enable delivery. The webhook URL is never
//! stored in a container or printed by this crate. Notification failures are
//! reported to stderr and do not change the caller's codec result.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

const WEBHOOK_ENV: &str = "VPCC_DISCORD_WEBHOOK_URL";

#[derive(Debug, Clone)]
pub struct RunContext {
    pub run_id: String,
    pub operation: &'static str,
    pub input_path: String,
    pub output_path: String,
}

impl RunContext {
    pub fn new(
        operation: &'static str,
        input_path: impl Into<String>,
        output_path: impl Into<String>,
    ) -> Self {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        Self {
            run_id: format!("{operation}-{}-{now_ms}", std::process::id()),
            operation,
            input_path: input_path.into(),
            output_path: output_path.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionDetails {
    pub elapsed: Duration,
    pub total_bytes: Option<u64>,
    pub details: Vec<(String, String)>,
}

impl CompletionDetails {
    pub fn new(elapsed: Duration) -> Self {
        Self {
            elapsed,
            total_bytes: None,
            details: Vec::new(),
        }
    }

    pub fn with_total_bytes(mut self, total_bytes: u64) -> Self {
        self.total_bytes = Some(total_bytes);
        self
    }

    pub fn add(mut self, label: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.push((label.into(), value.into()));
        self
    }
}

pub struct DiscordNotifier {
    webhook_url: Option<String>,
    context: RunContext,
}

impl DiscordNotifier {
    pub fn from_env(context: RunContext) -> Self {
        // Explicit local-only execution must also suppress dotenv fallback.
        if std::env::var("VPCC_DISCORD_DISABLED").as_deref() == Ok("1") {
            return Self::disabled(context);
        }
        let webhook_url = std::env::var(WEBHOOK_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(webhook_url_from_dotenv);
        Self {
            webhook_url,
            context,
        }
    }

    pub fn disabled(context: RunContext) -> Self {
        Self {
            webhook_url: None,
            context,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.webhook_url.is_some()
    }

    pub fn run_started(&self) {
        self.deliver(
            "started",
            None,
            vec![
                ("input", self.context.input_path.clone()),
                ("output", self.context.output_path.clone()),
            ],
        );
    }

    pub fn stage_changed(&self, stage: &str) {
        self.deliver("progress", Some(stage), Vec::new());
    }

    pub fn run_completed(&self, completion: CompletionDetails) {
        let mut details = completion.details;
        details.push(("elapsed".to_string(), format_duration(completion.elapsed)));
        if let Some(total_bytes) = completion.total_bytes {
            details.push(("container bytes".to_string(), total_bytes.to_string()));
        }
        details.push(("output".to_string(), self.context.output_path.clone()));
        self.deliver_owned("completed", None, details);
    }

    pub fn run_failed(&self, stage: &str, error: &str, elapsed: Duration) {
        self.deliver(
            "failed",
            Some(stage),
            vec![
                ("elapsed", format_duration(elapsed)),
                ("error", truncate(error, 900)),
            ],
        );
    }

    fn deliver(&self, outcome: &str, stage: Option<&str>, details: Vec<(&str, String)>) {
        let details = details
            .into_iter()
            .map(|(label, value)| (label.to_string(), value))
            .collect();
        self.deliver_owned(outcome, stage, details);
    }

    fn deliver_owned(&self, outcome: &str, stage: Option<&str>, details: Vec<(String, String)>) {
        let Some(webhook_url) = self.webhook_url.as_deref() else {
            return;
        };
        let content = format_message(&self.context, outcome, stage, &details);
        let payload = WebhookPayload { content };
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(10)))
            .build();
        let agent: ureq::Agent = config.into();
        let result = agent.post(webhook_url).send_json(&payload);
        if result.is_err() {
            // Do not include the transport error: it can contain the webhook URL.
            eprintln!("Discord notification delivery failed ({outcome}); codec run continues.");
        }
    }
}

fn webhook_url_from_dotenv() -> Option<String> {
    let current_dir = std::env::current_dir().ok()?;
    for directory in current_dir.ancestors() {
        let dotenv_path = directory.join(".env");
        let contents = match std::fs::read_to_string(dotenv_path) {
            Ok(contents) => contents,
            Err(_) => continue,
        };
        if let Some(value) = dotenv_value(&contents, WEBHOOK_ENV) {
            return Some(value);
        }
    }
    None
}

fn dotenv_value(contents: &str, key: &str) -> Option<String> {
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((candidate_key, candidate_value)) = line.split_once('=') else {
            continue;
        };
        if candidate_key.trim() != key {
            continue;
        }
        let value = candidate_value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .or_else(|| {
                value
                    .strip_prefix('\'')
                    .and_then(|value| value.strip_suffix('\''))
            })
            .unwrap_or(value)
            .trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

#[derive(Serialize)]
struct WebhookPayload {
    content: String,
}

fn format_message(
    context: &RunContext,
    outcome: &str,
    stage: Option<&str>,
    details: &[(String, String)],
) -> String {
    let mut lines = vec![format!(
        "**VPCC {} {}** (`{}`)",
        context.operation, outcome, context.run_id
    )];
    if let Some(stage) = stage {
        lines.push(format!("Stage: {stage}"));
    }
    for (label, value) in details {
        lines.push(format!("{label}: {}", truncate(value, 700)));
    }
    truncate(&lines.join("\n"), 1_900)
}

fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{}s", duration.as_secs_f64().round())
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let prefix: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn environment_disable_overrides_configured_webhook() {
        // Isolate environment variables from the parallel test process.
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::environment_disable_child"])
            .env("VPCC_TEST_DISABLE_CHILD", "1")
            .env("VPCC_DISCORD_DISABLED", "1")
            .env(WEBHOOK_ENV, "http://127.0.0.1:1/not-used")
            .status().unwrap();
        assert!(result.success());
    }

    #[test]
    fn environment_disable_child() {
        if std::env::var("VPCC_TEST_DISABLE_CHILD").as_deref() == Ok("1") {
            let notifier = DiscordNotifier::from_env(RunContext::new("encode", "in", "out"));
            assert!(!notifier.is_enabled());
        }
    }

    #[test]
    fn disabled_notifier_accepts_lifecycle_events() {
        let notifier = DiscordNotifier::disabled(RunContext::new("encode", "in.ply", "runs/test"));
        assert!(!notifier.is_enabled());
        notifier.run_started();
        notifier.stage_changed("packing raster");
        notifier.run_completed(CompletionDetails::new(Duration::from_secs(1)));
        notifier.run_failed("writing", "synthetic error", Duration::from_secs(1));
    }

    #[test]
    fn formatted_message_has_context_and_respects_discord_limit() {
        let context = RunContext {
            run_id: "encode-1".to_string(),
            operation: "encode",
            input_path: "in.ply".to_string(),
            output_path: "out".to_string(),
        };
        let message = format_message(
            &context,
            "failed",
            Some("writing"),
            &[("error".to_string(), "x".repeat(3_000))],
        );
        assert!(message.contains("encode-1"));
        assert!(message.contains("Stage: writing"));
        assert!(message.chars().count() <= 1_900);
    }

    #[test]
    fn dotenv_parser_reads_quoted_and_exported_values() {
        let contents =
            "# local settings\nexport VPCC_DISCORD_WEBHOOK_URL='https://example.test/webhook'\n";
        assert_eq!(
            dotenv_value(contents, WEBHOOK_ENV).as_deref(),
            Some("https://example.test/webhook")
        );
    }

    #[test]
    fn webhook_notifier_posts_json_to_a_local_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request_bytes = Vec::new();
            let mut buffer = [0u8; 1_024];
            loop {
                let bytes_read = stream.read(&mut buffer).unwrap();
                if bytes_read == 0 {
                    break;
                }
                request_bytes.extend_from_slice(&buffer[..bytes_read]);
                let request = String::from_utf8_lossy(&request_bytes);
                let Some(header_end) = request.find("\r\n\r\n") else {
                    continue;
                };
                let content_length = request[..header_end]
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then_some(value.trim())
                    })
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0);
                if request_bytes.len() >= header_end + 4 + content_length {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            stream
                .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            request
        });

        let notifier = DiscordNotifier {
            webhook_url: Some(format!("http://{address}/webhook")),
            context: RunContext {
                run_id: "decode-1".to_string(),
                operation: "decode",
                input_path: "container".to_string(),
                output_path: "decoded.ply".to_string(),
            },
        };
        notifier.run_started();

        let request = receiver.join().unwrap();
        assert!(request.starts_with("POST /webhook HTTP/1.1"));
        assert!(request.contains("content"));
        assert!(request.contains("VPCC decode started"));
    }
}
