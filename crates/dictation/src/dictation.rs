//! Voice dictation: record the microphone with ffmpeg, transcribe the audio
//! with Google Gemini, and hand the text back to the caller.
//!
//! This crate is intentionally GPUI-free so it can be built and tested in
//! isolation (`cargo build -p dictation`). The terminal panel drives it and is
//! responsible for inserting the returned text into the active terminal.
//!
//! Ported from the TypeScript dictation feature in the `dev-environment-manager`
//! VS Code extension (`plugin/src/workflows/dictation/*`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use futures::AsyncReadExt as _;
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use serde::Deserialize;
use smol::io::AsyncWriteExt as _;
use tempfile::TempDir;
use util::command::{Stdio, new_command};

const GEMINI_HOST: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_MODEL: &str = "gemini-2.5-flash";
const SAMPLE_RATE: u32 = 16_000;
/// How long to wait after spawning ffmpeg before we trust the backend opened.
const SPAWN_CONFIRM: Duration = Duration::from_millis(400);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_ATTEMPTS: usize = 3;

/// Human-readable language names keyed by the code used in the UI/settings.
/// `"auto"` (or any unknown key) lets Gemini detect the language.
const LANG_NAMES: &[(&str, &str)] = &[
    ("uk", "Ukrainian"),
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pl", "Polish"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("cs", "Czech"),
    ("ja", "Japanese"),
    ("zh", "Chinese"),
    ("ko", "Korean"),
];

/// An in-flight recording: the live ffmpeg process plus the temp dir holding
/// the WAV. Dropping it kills ffmpeg and removes the temp dir.
pub struct Recording {
    child: util::command::Child,
    wav_path: PathBuf,
    // Kept alive so the WAV file survives until transcription; dropped (and the
    // directory removed) once the caller drops the `PathBuf` scope after use.
    _dir: TempDir,
}

/// ffmpeg capture backend. We try PulseAudio/PipeWire first, then ALSA.
#[derive(Clone, Copy)]
enum Backend {
    Pulse,
    Alsa,
}

impl Backend {
    fn as_ffmpeg_input(self) -> &'static str {
        match self {
            Backend::Pulse => "pulse",
            Backend::Alsa => "alsa",
        }
    }
}

/// Start recording the default microphone. Spawns ffmpeg writing a mono 16-bit
/// PCM WAV. Tries PulseAudio then ALSA; errors if neither opens.
///
/// ffmpeg is spawned with a piped stdin (no `-nostdin`) so [`stop_recording`]
/// can send `q` for a clean WAV finalization instead of a signal.
pub async fn start_recording() -> Result<Recording> {
    if !is_bin_on_path("ffmpeg") {
        bail!("ffmpeg is required for voice dictation. Install it, e.g. `sudo apt install ffmpeg`.");
    }

    let dir = tempfile::Builder::new()
        .prefix("zed-dictation-")
        .tempdir()
        .context("creating dictation temp dir")?;
    let wav_path = dir.path().join("capture.wav");

    let child = match spawn_ffmpeg(Backend::Pulse, &wav_path).await {
        Some(child) => child,
        None => spawn_ffmpeg(Backend::Alsa, &wav_path)
            .await
            .ok_or_else(|| {
                anyhow!("No microphone input found (tried PulseAudio and ALSA). Ensure a default input device is configured.")
            })?,
    };

    Ok(Recording {
        child,
        wav_path,
        _dir: dir,
    })
}

/// Stop the recording gracefully and return the finalized WAV path.
///
/// Sends `q` to ffmpeg's stdin (its interactive quit) so the WAV header is
/// written, then waits for exit. Falls back to a kill if `q` doesn't take.
pub async fn stop_recording(mut rec: Recording) -> Result<PathBuf> {
    if let Some(mut stdin) = rec.child.stdin.take() {
        // Best-effort: ffmpeg quits cleanly on `q`. Ignore write errors — it may
        // have already exited, in which case the WAV is already finalized.
        let _ = stdin.write_all(b"q").await;
        let _ = stdin.flush().await;
    }

    // Give ffmpeg a moment to finalize, then hard-stop if still alive.
    let status = smol::future::or(async { Some(rec.child.status().await) }, async {
        smol::Timer::after(Duration::from_secs(5)).await;
        None
    })
    .await;

    if status.is_none() {
        let _ = rec.child.kill();
        let _ = rec.child.status().await;
    }

    let wav = rec.wav_path.clone();
    if !wav_is_ready(&wav) {
        // `rec._dir` drops here → the temp dir is removed. No leak.
        bail!("Recording produced no audio — try speaking a little longer.");
    }
    // Detach the auto-delete guard so the WAV outlives this function — the
    // caller transcribes it next and then calls `cleanup` to remove the dir
    // (process exit reclaims /tmp otherwise).
    std::mem::forget(rec._dir);
    Ok(wav)
}

/// Remove a finalized WAV and its temp directory after transcription.
pub fn cleanup(wav_path: &Path) {
    if let Some(parent) = wav_path.parent() {
        let _ = std::fs::remove_dir_all(parent);
    }
}

/// Transcribe a finalized WAV with Gemini. Uploads via the Files API (streamed,
/// no full-file buffering), then calls `generateContent`.
pub async fn transcribe_wav(
    http: &dyn HttpClient,
    api_key: &str,
    wav_path: &Path,
    language: &str,
) -> Result<String> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        bail!("GEMINI_API_KEY is not set — export it before using voice dictation.");
    }

    let file_uri = upload_to_files_api(http, api_key, wav_path).await?;
    let prompt = build_prompt(language);

    let body = serde_json::json!({
        "contents": [{
            "parts": [
                { "file_data": { "mime_type": "audio/wav", "file_uri": file_uri } },
                { "text": prompt },
            ],
        }],
    });
    let body = serde_json::to_string(&body)?;

    let uri = format!("{GEMINI_HOST}/v1beta/models/{DEFAULT_MODEL}:generateContent");
    let response = with_retry(|| {
        let body = body.clone();
        let uri = uri.clone();
        async move {
            let request = HttpRequest::builder()
                .method(Method::POST)
                .uri(uri)
                .header("Content-Type", "application/json")
                .header("x-goog-api-key", api_key)
                .body(AsyncBody::from(body))?;
            send_read(http, request).await
        }
    })
    .await?;

    let parsed: GenerateContentResponse = serde_json::from_str(&response)
        .with_context(|| format!("parsing Gemini response: {response}"))?;
    let text = parsed
        .candidates
        .into_iter()
        .next()
        .and_then(|c| c.content.parts.into_iter().next())
        .and_then(|p| p.text)
        .unwrap_or_default();
    let trimmed = text.trim().to_string();
    if trimmed.is_empty() {
        bail!("Gemini returned an empty transcription");
    }
    Ok(trimmed)
}

// ─── ffmpeg recording ────────────────────────────────────────────

async fn spawn_ffmpeg(backend: Backend, wav_path: &Path) -> Option<util::command::Child> {
    let mut command = new_command("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            backend.as_ffmpeg_input(),
            "-i",
            "default",
            "-ac",
            "1",
            "-ar",
        ])
        .arg(SAMPLE_RATE.to_string())
        .args(["-sample_fmt", "s16"])
        .arg(wav_path)
        // Scrub loader vars the parent may have inherited from a bundled
        // runtime (e.g. a Flatpak wrapper sets LD_LIBRARY_PATH to its own libs,
        // which shadow system libraries and make system ffmpeg crash on a
        // symbol lookup). ffmpeg is a system binary and must run against system
        // libraries.
        .env_remove("LD_LIBRARY_PATH")
        .env_remove("LD_PRELOAD")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);

    let mut child = command.spawn().ok()?;

    // If ffmpeg fails instantly (unknown backend / no device) it exits within
    // the confirm window. If it's still alive after, trust the backend opened.
    let exited = smol::future::or(async { Some(child.status().await) }, async {
        smol::Timer::after(SPAWN_CONFIRM).await;
        None
    })
    .await;

    match exited {
        None => Some(child),   // still running → good
        Some(_) => None,       // exited early → try the next backend
    }
}

fn wav_is_ready(wav_path: &Path) -> bool {
    // A valid WAV has at least a 44-byte header; anything smaller means no audio.
    std::fs::metadata(wav_path)
        .map(|m| m.len() > 44)
        .unwrap_or(false)
}

fn is_bin_on_path(bin: &str) -> bool {
    let Ok(path) = std::env::var("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| dir.join(bin).is_file())
}

// ─── Gemini HTTP ─────────────────────────────────────────────────

async fn upload_to_files_api(
    http: &dyn HttpClient,
    api_key: &str,
    wav_path: &Path,
) -> Result<String> {
    let size = std::fs::metadata(wav_path)
        .with_context(|| format!("stat {}", wav_path.display()))?
        .len();

    let response = with_retry(|| async move {
        // A fresh reader per attempt: a stream can only be consumed once.
        let file = async_fs::File::open(wav_path)
            .await
            .with_context(|| format!("open {}", wav_path.display()))?;
        let request = HttpRequest::builder()
            .method(Method::POST)
            .uri(format!("{GEMINI_HOST}/upload/v1beta/files?uploadType=media"))
            .header("Content-Type", "audio/wav")
            .header("Content-Length", size)
            .header("x-goog-api-key", api_key)
            .body(AsyncBody::from_reader(file))?;
        send_read(http, request).await
    })
    .await?;

    let parsed: FilesApiResponse = serde_json::from_str(&response)
        .with_context(|| format!("parsing Files API response: {response}"))?;
    parsed
        .file
        .and_then(|f| f.uri)
        .filter(|u| !u.is_empty())
        .ok_or_else(|| anyhow!("Gemini Files API did not return a file URI"))
}

/// Send a request and read the whole body to a `String`, mapping non-2xx to a
/// retryable/terminal error via [`HttpError`].
async fn send_read(http: &dyn HttpClient, request: http_client::Request<AsyncBody>) -> Result<String> {
    let response = http
        .send(request)
        .await
        .map_err(|e| HttpError::Transport(e.to_string()))?;
    let status = response.status();
    let mut body = String::new();
    response
        .into_body()
        .read_to_string(&mut body)
        .await
        .map_err(|e| HttpError::Transport(e.to_string()))?;

    if status.is_success() {
        Ok(body)
    } else {
        let message = extract_error_message(&body).unwrap_or_else(|| format!("HTTP {status}"));
        Err(HttpError::Status {
            code: status.as_u16(),
            message,
        }
        .into())
    }
}

/// Retry `fn` up to [`MAX_ATTEMPTS`] with exponential backoff (1s, 2s, 4s) on
/// 429/5xx and transport errors. Non-retryable statuses fail immediately.
async fn with_retry<F, Fut>(mut make: F) -> Result<String>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=MAX_ATTEMPTS {
        // Per-attempt timeout guard.
        let result = smol::future::or(async { Some(make().await) }, async {
            smol::Timer::after(REQUEST_TIMEOUT).await;
            None
        })
        .await;

        match result {
            Some(Ok(body)) => return Ok(body),
            Some(Err(err)) => {
                let retryable = err
                    .downcast_ref::<HttpError>()
                    .map(HttpError::is_retryable)
                    .unwrap_or(false);
                if !retryable || attempt == MAX_ATTEMPTS {
                    return Err(err);
                }
                last_err = Some(err);
            }
            None => {
                last_err = Some(anyhow!("Gemini request timed out after {REQUEST_TIMEOUT:?}"));
                if attempt == MAX_ATTEMPTS {
                    break;
                }
            }
        }

        let backoff = Duration::from_secs(1 << (attempt - 1));
        smol::Timer::after(backoff).await;
    }
    Err(last_err.unwrap_or_else(|| anyhow!("Gemini request failed")))
}

#[derive(Debug)]
enum HttpError {
    Transport(String),
    Status { code: u16, message: String },
}

impl HttpError {
    fn is_retryable(&self) -> bool {
        match self {
            HttpError::Transport(_) => true,
            HttpError::Status { code, .. } => *code == 429 || (500..600).contains(code),
        }
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::Transport(m) => write!(f, "network error: {m}"),
            HttpError::Status { message, .. } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for HttpError {}

fn extract_error_message(body: &str) -> Option<String> {
    let parsed: ErrorResponse = serde_json::from_str(body).ok()?;
    parsed.error.and_then(|e| e.message).filter(|m| !m.is_empty())
}

fn build_prompt(language: &str) -> String {
    let name = LANG_NAMES
        .iter()
        .find(|(code, _)| *code == language)
        .map(|(_, name)| *name);
    match name {
        Some(name) => format!(
            "Transcribe this audio recording exactly as spoken. Return only the \
             transcribed text, nothing else. The spoken language is {name}. Transcribe in {name}."
        ),
        None => "Transcribe this audio recording exactly as spoken. Return only the \
                 transcribed text, nothing else. Preserve the original language — do not translate."
            .to_string(),
    }
}

// ─── Gemini response shapes ──────────────────────────────────────

#[derive(Deserialize)]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Content,
}

#[derive(Deserialize)]
struct Content {
    #[serde(default)]
    parts: Vec<Part>,
}

#[derive(Deserialize)]
struct Part {
    text: Option<String>,
}

#[derive(Deserialize)]
struct FilesApiResponse {
    file: Option<FileInfo>,
}

#[derive(Deserialize)]
struct FileInfo {
    uri: Option<String>,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: Option<ErrorBody>,
}

#[derive(Deserialize)]
struct ErrorBody {
    message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prompt_auto_preserves_language() {
        let p = build_prompt("auto");
        assert!(p.contains("Preserve the original language"));
    }

    #[test]
    fn build_prompt_known_language_names_it() {
        let p = build_prompt("uk");
        assert!(p.contains("Ukrainian"));
        assert!(!p.contains("Preserve the original language"));
    }

    #[test]
    fn build_prompt_unknown_code_falls_back_to_auto() {
        let p = build_prompt("xx");
        assert!(p.contains("Preserve the original language"));
    }

    #[test]
    fn http_status_retryable_classification() {
        assert!(HttpError::Status { code: 429, message: String::new() }.is_retryable());
        assert!(HttpError::Status { code: 503, message: String::new() }.is_retryable());
        assert!(!HttpError::Status { code: 400, message: String::new() }.is_retryable());
        assert!(HttpError::Transport("boom".into()).is_retryable());
    }

    #[test]
    fn extract_error_message_reads_gemini_shape() {
        let body = r#"{"error":{"message":"bad key","code":401}}"#;
        assert_eq!(extract_error_message(body).as_deref(), Some("bad key"));
        assert_eq!(extract_error_message("not json"), None);
    }
}
