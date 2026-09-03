//! Voice dictation: record the microphone with ffmpeg, transcribe the audio
//! with Google Gemini, and hand the text back to the caller.
//!
//! This crate is intentionally GPUI-free so it can be built and tested in
//! isolation (`cargo build -p dictation`). The terminal panel drives it and is
//! responsible for inserting the returned text into the active terminal.
//!
//! Ported from the TypeScript dictation feature in the `dev-environment-manager`
//! VS Code extension (`plugin/src/workflows/dictation/*`).

// This crate is deliberately GPUI-free (see module docs above), so the
// executor-based timer clippy suggests as the replacement is unavailable here.
// The `smol::Timer::after` calls below are real wall-clock timeouts on
// production paths (ffmpeg lifecycle, stop lead-out, request timeout, retry
// backoff), not test timing, so the non-determinism concern does not apply.
#![allow(
    clippy::disallowed_methods,
    reason = "GPUI-free crate: gpui executor timer is unavailable; timers are production wall-clock timeouts"
)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use base64::Engine as _;
use futures::AsyncReadExt as _;
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use serde::Deserialize;
use smol::io::AsyncWriteExt as _;
use tempfile::TempDir;
use util::command::{Stdio, new_command};

const GEMINI_HOST: &str = "https://generativelanguage.googleapis.com";
const SAMPLE_RATE: u32 = 16_000;
const BYTES_PER_SAMPLE: u64 = 2;
const WAV_HEADER_BYTES: u64 = 44;
/// ffmpeg exits within tens of milliseconds when the capture backend itself is
/// unavailable, so this window only has to outlast process startup — it is not
/// waiting for the device to open.
const SPAWN_CONFIRM: Duration = Duration::from_millis(150);
/// Time granted to ffmpeg to drain what the sound server still holds before we
/// ask it to quit. Without it `q` tears the process down along with the audio
/// that was buffered but not yet read, which clips the last word spoken.
const LEAD_OUT: Duration = Duration::from_millis(350);
/// Recordings shorter than this are a mis-click, not speech; rejecting them
/// locally avoids a pointless upload and a confusing empty transcription.
const MIN_DURATION: Duration = Duration::from_secs(1);
/// A request to the Interactions API is capped at 20 MB including the ~33%
/// base64 expansion, so larger recordings go through the Files API instead.
const MAX_INLINE_WAV_BYTES: u64 = 14 * 1024 * 1024;
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

/// Which capture backend last worked, so only the first recording of a session
/// pays for probing one that is unavailable on this machine.
static WORKING_BACKEND: AtomicU8 = AtomicU8::new(Backend::UNKNOWN);

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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Backend {
    Pulse,
    Alsa,
}

impl Backend {
    const UNKNOWN: u8 = 0;

    fn as_ffmpeg_input(self) -> &'static str {
        match self {
            Backend::Pulse => "pulse",
            Backend::Alsa => "alsa",
        }
    }

    fn as_cached(self) -> u8 {
        match self {
            Backend::Pulse => 1,
            Backend::Alsa => 2,
        }
    }

    fn from_cached(value: u8) -> Option<Self> {
        match value {
            1 => Some(Backend::Pulse),
            2 => Some(Backend::Alsa),
            _ => None,
        }
    }
}

/// Start recording the default microphone. Spawns ffmpeg writing a mono 16-bit
/// PCM WAV, capped at `max_duration` so a forgotten recording cannot grow
/// unbounded. Tries PulseAudio then ALSA; errors if neither opens.
///
/// ffmpeg is spawned with a piped stdin (no `-nostdin`) so [`stop_recording`]
/// can send `q` for a clean WAV finalization instead of a signal.
pub async fn start_recording(max_duration: Duration) -> Result<Recording> {
    if !is_bin_on_path("ffmpeg") {
        bail!("ffmpeg is required for voice dictation. Install it, e.g. `sudo apt install ffmpeg`.");
    }

    let dir = tempfile::Builder::new()
        .prefix("zed-dictation-")
        .tempdir()
        .context("creating dictation temp dir")?;
    let wav_path = dir.path().join("capture.wav");

    let child = open_capture(&wav_path, max_duration).await.ok_or_else(|| {
        anyhow!(
            "No microphone input found (tried PulseAudio and ALSA). Ensure a default input device is configured."
        )
    })?;

    Ok(Recording {
        child,
        wav_path,
        _dir: dir,
    })
}

/// Stop the recording gracefully and return the finalized WAV path.
///
/// Waits out [`LEAD_OUT`] so the tail of the last word is captured, then sends
/// `q` to ffmpeg's stdin (its interactive quit) so the WAV header is written,
/// then waits for exit. Falls back to a kill if `q` doesn't take.
pub async fn stop_recording(mut rec: Recording) -> Result<PathBuf> {
    smol::Timer::after(LEAD_OUT).await;

    if let Some(mut stdin) = rec.child.stdin.take() {
        // Best-effort: ffmpeg quits cleanly on `q`. A write failure here means
        // it already exited (`-t` elapsed), in which case the WAV is finalized
        // and there is nothing to recover from.
        if let Err(err) = stdin.write_all(b"q").await {
            log::debug!("dictation: ffmpeg stdin already closed: {err}");
        } else if let Err(err) = stdin.flush().await {
            log::debug!("dictation: flushing ffmpeg stdin failed: {err}");
        }
    }

    // Give ffmpeg a moment to finalize, then hard-stop if still alive.
    let status = smol::future::or(async { Some(rec.child.status().await) }, async {
        smol::Timer::after(Duration::from_secs(5)).await;
        None
    })
    .await;

    if status.is_none() {
        log_teardown_failure(rec.child.kill(), "killing stalled ffmpeg");
        log_teardown_failure(rec.child.status().await, "reaping killed ffmpeg");
    }

    let wav = rec.wav_path.clone();
    if !wav_has_min_duration(&wav) {
        // The trusted backend may have gone away since it last worked, which
        // looks exactly like a too-short recording. Forget it so the next
        // attempt probes both backends again instead of failing forever.
        WORKING_BACKEND.store(Backend::UNKNOWN, Ordering::Relaxed);
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
        log_teardown_failure(std::fs::remove_dir_all(parent), "removing dictation temp dir");
    }
}

/// Transcribe a finalized WAV with Gemini.
///
/// Dedicated transcription models take the audio directly over the Interactions
/// API; any other model id falls back to `generateContent` with a prompt and
/// thinking disabled. Audio is sent inline whenever it fits, which keeps the
/// whole transcription to a single round trip.
pub async fn transcribe_wav(
    http: &dyn HttpClient,
    api_key: &str,
    wav_path: &Path,
    model: &str,
    language: &str,
) -> Result<String> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        bail!("No Gemini API key configured — add one under the Google language model provider.");
    }

    let audio = load_audio(http, api_key, wav_path).await?;
    if is_transcription_model(model) {
        transcribe_via_interactions(http, api_key, model, audio).await
    } else {
        transcribe_via_generate_content(http, api_key, model, language, audio).await
    }
}

// ─── ffmpeg recording ────────────────────────────────────────────

/// Open a capture stream, preferring the backend that already worked.
///
/// A backend proven this session is trusted without waiting out
/// [`SPAWN_CONFIRM`]: skipping that window is what lets recording begin the
/// moment the user asks for it. A backend that has since gone away surfaces as
/// a too-short recording, and [`stop_recording`] clears the cache so the next
/// attempt probes again.
async fn open_capture(wav_path: &Path, max_duration: Duration) -> Option<util::command::Child> {
    let cached = Backend::from_cached(WORKING_BACKEND.load(Ordering::Relaxed));
    if let Some(backend) = cached
        && let Some(child) = spawn_ffmpeg(backend, wav_path, max_duration, Confirm::No).await
    {
        return Some(child);
    }

    for backend in [Backend::Pulse, Backend::Alsa] {
        if let Some(child) = spawn_ffmpeg(backend, wav_path, max_duration, Confirm::Yes).await {
            WORKING_BACKEND.store(backend.as_cached(), Ordering::Relaxed);
            return Some(child);
        }
    }
    WORKING_BACKEND.store(Backend::UNKNOWN, Ordering::Relaxed);
    None
}

/// Whether to spend [`SPAWN_CONFIRM`] verifying that ffmpeg stayed up.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Confirm {
    Yes,
    No,
}

async fn spawn_ffmpeg(
    backend: Backend,
    wav_path: &Path,
    max_duration: Duration,
    confirm: Confirm,
) -> Option<util::command::Child> {
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
        .args(["-sample_fmt", "s16", "-t"])
        // A cap below the minimum would reject every recording it produced.
        .arg(max_duration.max(MIN_DURATION).as_secs().to_string())
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

    if confirm == Confirm::No {
        return Some(child);
    }

    // If ffmpeg fails instantly (unknown backend / no device) it exits within
    // the confirm window. If it's still alive after, trust the backend opened.
    let exited = smol::future::or(async { Some(child.status().await) }, async {
        smol::Timer::after(SPAWN_CONFIRM).await;
        None
    })
    .await;

    match exited {
        None => Some(child),
        Some(_) => None,
    }
}

/// Smallest WAV that still holds [`MIN_DURATION`] of mono 16-bit audio.
fn min_wav_bytes() -> u64 {
    WAV_HEADER_BYTES + u64::from(SAMPLE_RATE) * BYTES_PER_SAMPLE * MIN_DURATION.as_secs()
}

fn wav_has_min_duration(wav_path: &Path) -> bool {
    std::fs::metadata(wav_path)
        .map(|metadata| metadata.len() >= min_wav_bytes())
        .unwrap_or(false)
}

fn is_bin_on_path(bin: &str) -> bool {
    let Ok(path) = std::env::var("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| dir.join(bin).is_file())
}

/// Teardown steps whose failure must not mask the real error but should still
/// be visible in the log.
fn log_teardown_failure<T, E: std::fmt::Display>(
    result: std::result::Result<T, E>,
    context: &str,
) {
    if let Err(err) = result {
        log::warn!("dictation: {context}: {err}");
    }
}

// ─── Gemini HTTP ─────────────────────────────────────────────────

/// How the recording is handed to Gemini: inline when it fits in a request,
/// otherwise by reference after an upload.
enum Audio {
    Inline(String),
    Uploaded(String),
}

fn is_transcription_model(model: &str) -> bool {
    model.contains("transcribe")
}

async fn load_audio(http: &dyn HttpClient, api_key: &str, wav_path: &Path) -> Result<Audio> {
    let size = std::fs::metadata(wav_path)
        .with_context(|| format!("stat {}", wav_path.display()))?
        .len();

    if size > MAX_INLINE_WAV_BYTES {
        return Ok(Audio::Uploaded(
            upload_to_files_api(http, api_key, wav_path, size).await?,
        ));
    }

    let bytes = std::fs::read(wav_path).with_context(|| format!("read {}", wav_path.display()))?;
    Ok(Audio::Inline(
        base64::engine::general_purpose::STANDARD.encode(bytes),
    ))
}

/// Transcribe with a dedicated speech-to-text model. These models need no
/// prompt and detect the spoken language themselves.
async fn transcribe_via_interactions(
    http: &dyn HttpClient,
    api_key: &str,
    model: &str,
    audio: Audio,
) -> Result<String> {
    let body = serde_json::to_string(&serde_json::json!({
        "model": model,
        "input": [audio_input_part(&audio)],
        "generation_config": { "transcription_config": { "mode": "smart" } },
    }))?;

    let uri = format!("{GEMINI_HOST}/v1beta/interactions");
    let response = post_json(http, api_key, &uri, &body).await?;

    let parsed: InteractionsResponse = serde_json::from_str(&response)
        .with_context(|| format!("parsing Interactions response: {response}"))?;
    non_empty_transcription(parsed.text())
}

/// Transcribe with a general-purpose model. Thinking is switched off: it buys
/// nothing for a transcription and costs seconds of latency.
async fn transcribe_via_generate_content(
    http: &dyn HttpClient,
    api_key: &str,
    model: &str,
    language: &str,
    audio: Audio,
) -> Result<String> {
    let body = serde_json::to_string(&serde_json::json!({
        "contents": [{
            "parts": [
                audio_content_part(&audio),
                { "text": build_prompt(language) },
            ],
        }],
        "generationConfig": { "thinkingConfig": thinking_config(model) },
    }))?;

    let uri = format!("{GEMINI_HOST}/v1beta/models/{model}:generateContent");
    let response = post_json(http, api_key, &uri, &body).await?;

    let parsed: GenerateContentResponse = serde_json::from_str(&response)
        .with_context(|| format!("parsing Gemini response: {response}"))?;
    non_empty_transcription(parsed.text())
}

fn audio_input_part(audio: &Audio) -> serde_json::Value {
    match audio {
        Audio::Inline(data) => serde_json::json!({
            "type": "audio",
            "data": data,
            "mime_type": "audio/wav",
        }),
        Audio::Uploaded(uri) => serde_json::json!({
            "type": "audio",
            "uri": uri,
            "mime_type": "audio/wav",
        }),
    }
}

fn audio_content_part(audio: &Audio) -> serde_json::Value {
    match audio {
        Audio::Inline(data) => serde_json::json!({
            "inline_data": { "mime_type": "audio/wav", "data": data },
        }),
        Audio::Uploaded(uri) => serde_json::json!({
            "file_data": { "mime_type": "audio/wav", "file_uri": uri },
        }),
    }
}

/// Gemini 3 models take a thinking *level* and may misbehave on a budget; 2.5
/// models are the other way round.
fn thinking_config(model: &str) -> serde_json::Value {
    if model.starts_with("gemini-3") {
        serde_json::json!({ "thinkingLevel": "minimal" })
    } else {
        serde_json::json!({ "thinkingBudget": 0 })
    }
}

fn non_empty_transcription(text: Option<String>) -> Result<String> {
    let trimmed = text.unwrap_or_default().trim().to_string();
    if trimmed.is_empty() {
        bail!("Gemini returned an empty transcription");
    }
    Ok(trimmed)
}

async fn post_json(http: &dyn HttpClient, api_key: &str, uri: &str, body: &str) -> Result<String> {
    with_retry(|| {
        let body = body.to_string();
        let uri = uri.to_string();
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
    .await
}

async fn upload_to_files_api(
    http: &dyn HttpClient,
    api_key: &str,
    wav_path: &Path,
    size: u64,
) -> Result<String> {
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
        .and_then(|file| file.uri)
        .filter(|uri| !uri.is_empty())
        .ok_or_else(|| anyhow!("Gemini Files API did not return a file URI"))
}

/// Send a request and read the whole body to a `String`, mapping non-2xx to a
/// retryable/terminal error via [`HttpError`].
async fn send_read(
    http: &dyn HttpClient,
    request: http_client::Request<AsyncBody>,
) -> Result<String> {
    let response = http
        .send(request)
        .await
        .map_err(|err| HttpError::Transport(err.to_string()))?;
    let status = response.status();
    let mut body = String::new();
    response
        .into_body()
        .read_to_string(&mut body)
        .await
        .map_err(|err| HttpError::Transport(err.to_string()))?;

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

/// Retry `make` up to [`MAX_ATTEMPTS`] with exponential backoff (1s, 2s, 4s) on
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
            HttpError::Transport(message) => write!(f, "network error: {message}"),
            HttpError::Status { message, .. } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for HttpError {}

fn extract_error_message(body: &str) -> Option<String> {
    let parsed: ErrorResponse = serde_json::from_str(body).ok()?;
    parsed
        .error
        .and_then(|error| error.message)
        .filter(|message| !message.is_empty())
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
struct InteractionsResponse {
    #[serde(default)]
    output: Vec<InteractionsOutput>,
    #[serde(default)]
    text: Option<String>,
}

impl InteractionsResponse {
    fn text(self) -> Option<String> {
        self.text.or_else(|| {
            self.output
                .into_iter()
                .find_map(|output| output.text.or(output.transcription))
        })
    }
}

#[derive(Deserialize)]
struct InteractionsOutput {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    transcription: Option<String>,
}

#[derive(Deserialize)]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
}

impl GenerateContentResponse {
    fn text(self) -> Option<String> {
        self.candidates
            .into_iter()
            .next()
            .and_then(|candidate| candidate.content.parts.into_iter().next())
            .and_then(|part| part.text)
    }
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
        let prompt = build_prompt("auto");
        assert!(prompt.contains("Preserve the original language"));
    }

    #[test]
    fn build_prompt_known_language_names_it() {
        let prompt = build_prompt("uk");
        assert!(prompt.contains("Ukrainian"));
        assert!(!prompt.contains("Preserve the original language"));
    }

    #[test]
    fn build_prompt_unknown_code_falls_back_to_auto() {
        let prompt = build_prompt("xx");
        assert!(prompt.contains("Preserve the original language"));
    }

    #[test]
    fn http_status_retryable_classification() {
        assert!(
            HttpError::Status {
                code: 429,
                message: String::new()
            }
            .is_retryable()
        );
        assert!(
            HttpError::Status {
                code: 503,
                message: String::new()
            }
            .is_retryable()
        );
        assert!(
            !HttpError::Status {
                code: 400,
                message: String::new()
            }
            .is_retryable()
        );
        assert!(HttpError::Transport("boom".into()).is_retryable());
    }

    #[test]
    fn extract_error_message_reads_gemini_shape() {
        let body = r#"{"error":{"message":"bad key","code":401}}"#;
        assert_eq!(extract_error_message(body).as_deref(), Some("bad key"));
        assert_eq!(extract_error_message("not json"), None);
    }

    #[test]
    fn transcription_models_take_the_interactions_path() {
        assert!(is_transcription_model("gemini-3.5-transcribe"));
        assert!(is_transcription_model("gemini-3.5-transcribe-live"));
        assert!(!is_transcription_model("gemini-flash-latest"));
        assert!(!is_transcription_model("gemini-2.5-flash"));
    }

    #[test]
    fn inline_audio_is_sent_as_data_and_uploaded_audio_as_uri() {
        let inline = audio_input_part(&Audio::Inline("QUJD".into()));
        assert_eq!(inline["type"], "audio");
        assert_eq!(inline["data"], "QUJD");
        assert_eq!(inline["mime_type"], "audio/wav");
        assert!(inline.get("uri").is_none());

        let uploaded = audio_input_part(&Audio::Uploaded("files/abc".into()));
        assert_eq!(uploaded["uri"], "files/abc");
        assert!(uploaded.get("data").is_none());
    }

    #[test]
    fn generate_content_parts_match_the_audio_source() {
        let inline = audio_content_part(&Audio::Inline("QUJD".into()));
        assert_eq!(inline["inline_data"]["data"], "QUJD");

        let uploaded = audio_content_part(&Audio::Uploaded("files/abc".into()));
        assert_eq!(uploaded["file_data"]["file_uri"], "files/abc");
    }

    #[test]
    fn thinking_is_disabled_the_way_each_model_family_expects() {
        let gemini_3 = thinking_config("gemini-3.5-flash");
        assert_eq!(gemini_3["thinkingLevel"], "minimal");
        assert!(gemini_3.get("thinkingBudget").is_none());

        let gemini_25 = thinking_config("gemini-2.5-flash");
        assert_eq!(gemini_25["thinkingBudget"], 0);
        assert!(gemini_25.get("thinkingLevel").is_none());
    }

    #[test]
    fn min_wav_bytes_covers_one_second_of_mono_16bit_audio() {
        assert_eq!(min_wav_bytes(), 44 + 32_000);
    }

    #[test]
    fn interactions_response_reads_both_shapes() {
        let flat: InteractionsResponse =
            serde_json::from_str(r#"{"text":"привіт"}"#).expect("flat shape parses");
        assert_eq!(flat.text().as_deref(), Some("привіт"));

        let nested: InteractionsResponse =
            serde_json::from_str(r#"{"output":[{"transcription":"привіт"}]}"#)
                .expect("nested shape parses");
        assert_eq!(nested.text().as_deref(), Some("привіт"));

        let empty: InteractionsResponse =
            serde_json::from_str("{}").expect("empty shape parses");
        assert!(empty.text().is_none());
    }

    #[test]
    fn empty_transcription_is_rejected() {
        assert!(non_empty_transcription(Some("  ".into())).is_err());
        assert!(non_empty_transcription(None).is_err());
        assert_eq!(
            non_empty_transcription(Some("  привіт  ".into())).expect("non-empty text is kept"),
            "привіт"
        );
    }

    #[test]
    fn backend_cache_round_trips() {
        for backend in [Backend::Pulse, Backend::Alsa] {
            assert_eq!(Backend::from_cached(backend.as_cached()), Some(backend));
        }
        assert!(Backend::from_cached(Backend::UNKNOWN).is_none());
    }
}
