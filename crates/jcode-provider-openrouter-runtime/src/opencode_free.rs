//! Anonymous OpenCode Zen free tier.
//!
//! The free lane is not an OpenAI-compatible endpoint in the ordinary sense:
//! it fingerprints the TLS ClientHello, insists on the OpenCode CLI's
//! correlation headers, and refuses a body that is not shaped like an agent
//! request (`stream: true`, `bash` + `read` tools, `<env>` in the system
//! prompt). A normal `reqwest`/rustls request is refused even with the right
//! headers, so this module carries a small BoringSSL HTTP/1.1 client whose
//! ClientHello matches Bun's.
//!
//! Rotation follows the community adapters: lanes (egress proxies or real
//! account keys) are chosen by `sticky`, `round-robin` or `lowest-usage`, a
//! refused lane cools down, and a gate rejection re-mints the session id.

use super::*;
use anyhow::{Context, Result};
use base64::Engine as _;
use bytes::Bytes;
use futures::StreamExt;
use http_body_util::{BodyStream, Full};
use hyper::Request;
use hyper_util::rt::TokioIo;
use jcode_message_types::ConnectionPhase;
use jcode_provider_openrouter::stream::OpenRouterStream;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

const API_HOST: &str = "opencode.ai";
const SESSION_MEMO_LIMIT: usize = 1024;
const MAX_ERROR_BODY: usize = 4 * 1024;
const MAX_CATALOG_BODY: usize = 1024 * 1024;
const BASE62: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Strategy {
    Sticky,
    RoundRobin,
    #[default]
    LowestUsage,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct FileConfig {
    #[serde(default)]
    strategy: Strategy,
    #[serde(default)]
    max_attempts: Option<usize>,
    #[serde(default)]
    cooldown_secs: Option<u64>,
    #[serde(default)]
    accounts: Vec<AccountConfig>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct AccountConfig {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    proxy: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
}

struct Lane {
    name: String,
    proxy: Option<String>,
    api_key: String,
    usage: AtomicU64,
    failures: AtomicU64,
    last_used_ms: AtomicU64,
    cooldown_until: Mutex<Option<Instant>>,
}

impl Lane {
    fn new(name: String, proxy: Option<String>, api_key: String) -> Self {
        Self {
            name,
            proxy,
            api_key,
            usage: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            last_used_ms: AtomicU64::new(0),
            cooldown_until: Mutex::new(None),
        }
    }

    fn healthy(&self, now: Instant) -> bool {
        self.cooldown_until
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_none_or(|until| until <= now)
    }

    /// Double the cooldown per consecutive refusal, capped at five minutes, so
    /// a spent egress is retried less often instead of being hammered.
    fn cool_down(&self, base: Duration) {
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        let factor = 1u32 << (failures.min(6) - 1) as u32;
        let backoff = base.saturating_mul(factor).min(Duration::from_secs(300));
        let until = Instant::now() + backoff;
        *self
            .cooldown_until
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(until);
    }

    fn mark_success(&self) {
        self.failures.store(0, Ordering::Relaxed);
        self.clear_cooldown();
    }

    fn clear_cooldown(&self) {
        *self
            .cooldown_until
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }
}

struct SessionStore {
    map: HashMap<String, String>,
    order: VecDeque<String>,
}

impl SessionStore {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get_or_mint(&mut self, signal: &str, fresh: bool) -> String {
        if !fresh && let Some(session) = self.map.get(signal) {
            return session.clone();
        }

        let session = mint_session_id(signal);
        if self
            .map
            .insert(signal.to_string(), session.clone())
            .is_none()
        {
            self.order.push_back(signal.to_string());
        }
        while self.order.len() > SESSION_MEMO_LIMIT {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        session
    }
}

/// Match the CLI's chat-completions body: `max_completion_tokens`, `store:
/// false`, and an explicit usage-bearing stream. Fields that only OpenRouter
/// understands are dropped before the request reaches the Zen gate.
pub(super) fn normalize_free_body(body: &mut Value) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    for key in ["provider", "transforms", "models", "route", "plugins"] {
        object.remove(key);
    }
    let max_tokens = object
        .remove("max_completion_tokens")
        .or_else(|| object.remove("max_output_tokens"))
        .or_else(|| object.remove("max_tokens"));
    if let Some(max_tokens) = max_tokens {
        object.insert("max_completion_tokens".to_string(), max_tokens);
    }
    object.insert("stream".to_string(), Value::Bool(true));
    object.insert("store".to_string(), Value::Bool(false));
    object.insert(
        "stream_options".to_string(),
        json!({ "include_usage": true }),
    );
}

pub struct OpenCodeFreeRuntime {
    connector: boring::ssl::SslConnector,
    lanes: Vec<Arc<Lane>>,
    strategy: Strategy,
    max_attempts: usize,
    cooldown: Duration,
    cursor: AtomicUsize,
    sessions: Mutex<SessionStore>,
}

static RUNTIME: OnceLock<std::result::Result<OpenCodeFreeRuntime, String>> = OnceLock::new();

pub fn runtime() -> Result<&'static OpenCodeFreeRuntime> {
    RUNTIME
        .get_or_init(|| OpenCodeFreeRuntime::new().map_err(|error| format!("{error:#}")))
        .as_ref()
        .map_err(|error| anyhow::anyhow!("OpenCode Free transport unavailable: {error}"))
}

/// Per-process lane list from the environment. An explicit `OPENCODE_FREE_LANES`
/// (comma-separated proxy URLs, or `none`/`direct`) replaces the config file so
/// each instance can pick its own egress; `OPENCODE_FREE_PROXY` is the
/// single-lane form. Unset both to use `~/.jcode/opencode-free.toml`.
fn env_accounts() -> Option<Vec<AccountConfig>> {
    let api_key = std::env::var("OPENCODE_FREE_API_KEY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "public".to_string());
    if let Ok(raw) = std::env::var("OPENCODE_FREE_LANES") {
        let lanes = raw
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        if !lanes.is_empty() {
            return Some(
                lanes
                    .into_iter()
                    .enumerate()
                    .map(|(index, proxy)| AccountConfig {
                        name: Some(format!("env-{}", index + 1)),
                        proxy: Some(proxy.to_string()),
                        api_key: Some(api_key.clone()),
                    })
                    .collect(),
            );
        }
    }
    std::env::var("OPENCODE_FREE_PROXY")
        .ok()
        .map(|proxy| AccountConfig {
            name: Some("env".to_string()),
            proxy: Some(proxy),
            api_key: Some(api_key),
        })
        .map(|account| vec![account])
}

impl OpenCodeFreeRuntime {
    fn new() -> Result<Self> {
        let config = load_config();
        let mut accounts = env_accounts().unwrap_or(config.accounts);
        if accounts.is_empty() {
            let api_key = std::env::var("OPENCODE_FREE_API_KEY")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "public".to_string());
            accounts.push(AccountConfig {
                name: Some("default".to_string()),
                proxy: default_proxy(),
                api_key: Some(api_key),
            });
        }

        let lanes = accounts
            .into_iter()
            .enumerate()
            .map(|(index, account)| {
                let name = account
                    .name
                    .map(|name| name.trim().to_string())
                    .filter(|name| !name.is_empty())
                    .unwrap_or_else(|| format!("lane-{}", index + 1));
                let proxy = account.proxy.as_deref().and_then(resolve_proxy);
                let api_key = account
                    .api_key
                    .map(|key| key.trim().to_string())
                    .filter(|key| !key.is_empty())
                    .or_else(|| {
                        std::env::var("OPENCODE_FREE_API_KEY")
                            .ok()
                            .map(|key| key.trim().to_string())
                            .filter(|key| !key.is_empty())
                    })
                    .unwrap_or_else(|| "public".to_string());
                Arc::new(Lane::new(name, proxy, api_key))
            })
            .collect::<Vec<_>>();

        let strategy = std::env::var("OPENCODE_FREE_STRATEGY")
            .ok()
            .and_then(|value| parse_strategy(&value))
            .unwrap_or(config.strategy);
        let max_attempts = std::env::var("OPENCODE_FREE_MAX_ATTEMPTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .or(config.max_attempts)
            .unwrap_or(3)
            .clamp(1, 10);
        let cooldown_secs = std::env::var("OPENCODE_FREE_COOLDOWN_SECS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .or(config.cooldown_secs)
            .unwrap_or(60);

        Ok(Self {
            connector: build_connector()?,
            lanes,
            strategy,
            max_attempts,
            cooldown: Duration::from_secs(cooldown_secs),
            cursor: AtomicUsize::new(0),
            sessions: Mutex::new(SessionStore::new()),
        })
    }

    fn session(&self, signal: &str, fresh: bool) -> String {
        self.sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_or_mint(signal, fresh)
    }

    fn pick_lane(&self, excluded: &[usize]) -> usize {
        let now = Instant::now();
        let mut candidates = (0..self.lanes.len())
            .filter(|index| !excluded.contains(index) && self.lanes[*index].healthy(now))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = (0..self.lanes.len())
                .filter(|index| !excluded.contains(index))
                .collect();
        }
        if candidates.is_empty() {
            candidates = (0..self.lanes.len()).collect();
        }

        match self.strategy {
            Strategy::Sticky => candidates[0],
            Strategy::RoundRobin => {
                let cursor = self.cursor.fetch_add(1, Ordering::Relaxed);
                candidates[cursor % candidates.len()]
            }
            Strategy::LowestUsage => *candidates
                .iter()
                .min_by_key(|index| {
                    let lane = &self.lanes[**index];
                    (
                        lane.usage.load(Ordering::Relaxed),
                        lane.last_used_ms.load(Ordering::Relaxed),
                    )
                })
                .expect("at least one lane"),
        }
    }

    pub async fn run(
        &'static self,
        api_base: String,
        request: Value,
        model: String,
        provider_pin: Arc<Mutex<Option<ProviderPin>>>,
        signal: String,
        tx: mpsc::Sender<Result<StreamEvent>>,
    ) {
        if tx
            .send(Ok(StreamEvent::ConnectionType {
                connection: "https/sse (opencode-free)".to_string(),
            }))
            .await
            .is_err()
        {
            return;
        }

        let mut excluded = Vec::new();
        let mut last_error = None;
        for attempt in 0..self.max_attempts {
            if tx.is_closed() {
                return;
            }
            let index = self.pick_lane(&excluded);
            let lane = Arc::clone(&self.lanes[index]);
            lane.last_used_ms.store(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                Ordering::Relaxed,
            );
            let session = self.session(&signal, attempt > 0);

            if tx
                .send(Ok(StreamEvent::ConnectionPhase {
                    phase: ConnectionPhase::SendingRequest,
                }))
                .await
                .is_err()
            {
                return;
            }

            match self.post(&lane, &session, &api_base, request.clone()).await {
                Ok(response) if response.status.is_success() => {
                    lane.usage.fetch_add(1, Ordering::Relaxed);
                    lane.mark_success();
                    if tx
                        .send(Ok(StreamEvent::ConnectionPhase {
                            phase: ConnectionPhase::WaitingForResponse,
                        }))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    self.stream(response, model, provider_pin, tx).await;
                    return;
                }
                Ok(response) => {
                    lane.usage.fetch_add(1, Ordering::Relaxed);
                    let status = response.status;
                    let retry_after = parse_retry_after(&response.headers);
                    let body = read_body_limited(response.body, MAX_ERROR_BODY).await;
                    let gate_rejected = matches!(status.as_u16(), 401 | 403 | 429);
                    if gate_rejected {
                        self.session(&signal, true);
                    }
                    let error = anyhow::anyhow!(
                        "OpenCode Free request failed\n  lane: {}\n  endpoint: {}/chat/completions\n  model: {}\n  status: {}\n  response: {}\nHint: the free lane refused this request. Check the model id, or add another lane/proxy to ~/.jcode/opencode-free.toml.",
                        lane.name,
                        api_base.trim_end_matches('/'),
                        model,
                        status,
                        body
                    );
                    if retryable_status(status) && attempt + 1 < self.max_attempts {
                        lane.cool_down(retry_after.unwrap_or(self.cooldown));
                        excluded.push(index);
                        last_error = Some(error);
                        jcode_base::logging::warn(&format!(
                            "OpenCode Free lane '{}' was refused with {status}; rotating",
                            lane.name
                        ));
                        continue;
                    }
                    let _ = tx.send(Err(error)).await;
                    return;
                }
                Err(error) => {
                    lane.usage.fetch_add(1, Ordering::Relaxed);
                    if attempt + 1 < self.max_attempts {
                        lane.cool_down(self.cooldown);
                        excluded.push(index);
                        last_error = Some(error);
                        jcode_base::logging::warn(&format!(
                            "OpenCode Free lane '{}' failed to connect; rotating",
                            lane.name
                        ));
                        continue;
                    }
                    let _ = tx.send(Err(error)).await;
                    return;
                }
            }
        }

        let _ = tx
            .send(Err(last_error.unwrap_or_else(|| {
                anyhow::anyhow!("OpenCode Free request failed after all lanes")
            })))
            .await;
    }

    async fn post(
        &self,
        lane: &Lane,
        session: &str,
        api_base: &str,
        body: Value,
    ) -> Result<FreeResponse> {
        let (host, port, path) = parse_api_endpoint(api_base, "chat/completions")?;
        let payload = serde_json::to_vec(&body).context("serializing OpenCode Free request")?;
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .header("host", &host)
            .header("authorization", format!("Bearer {}", lane.api_key))
            .header("content-type", "application/json")
            .header("accept", "*/*")
            .header("accept-encoding", "identity")
            .header("user-agent", opencode_user_agent())
            .header("x-opencode-client", "cli")
            .header("x-opencode-session", session)
            .header("x-opencode-session-id", session)
            .header("x-session-affinity", session)
            .header("x-session-id", session)
            .header("x-opencode-project", "global")
            .header("connection", "close")
            .body(Full::new(Bytes::from(payload)))
            .context("building OpenCode Free request")?;
        self.send(lane, &host, port, request).await
    }

    async fn send(
        &self,
        lane: &Lane,
        host: &str,
        port: u16,
        request: Request<Full<Bytes>>,
    ) -> Result<FreeResponse> {
        let tcp = connect(host, port, lane.proxy.as_deref()).await?;
        let config = self.connector.configure()?;
        let tls = tokio_boring::connect(config, host, tcp)
            .await
            .with_context(|| format!("OpenCode Free TLS handshake failed for {host}"))?;
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(tls))
            .await
            .context("OpenCode Free HTTP/1.1 handshake failed")?;
        tokio::spawn(async move {
            let _ = connection.await;
        });

        let response = sender
            .send_request(request)
            .await
            .context("sending OpenCode Free request")?;
        let (parts, body) = response.into_parts();
        let stream = BodyStream::new(body).filter_map(|frame| async move {
            match frame {
                Ok(frame) => frame.into_data().ok().map(Ok),
                Err(error) => Some(Err(anyhow::Error::from(error))),
            }
        });
        Ok(FreeResponse {
            status: parts.status,
            headers: parts.headers,
            body: Box::pin(stream),
        })
    }

    /// Fetch the live model roster over the same BoringSSL transport. A plain
    /// reqwest GET is refused by the same TLS gate as inference even though
    /// `GET /models` itself does not require the agent body.
    pub async fn fetch_models_json(&self, api_base: &str) -> Result<String> {
        let (host, port, path) = parse_api_endpoint(api_base, "models")?;
        let mut excluded = Vec::new();
        let mut last_error = None;
        for _ in 0..self.max_attempts.max(1) {
            let index = self.pick_lane(&excluded);
            let lane = &self.lanes[index];
            lane.last_used_ms.store(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                Ordering::Relaxed,
            );
            let request = Request::builder()
                .method("GET")
                .uri(&path)
                .header("host", &host)
                .header("authorization", format!("Bearer {}", lane.api_key))
                .header("accept", "application/json")
                .header("accept-encoding", "identity")
                .header("user-agent", opencode_user_agent())
                .header("x-opencode-client", "cli")
                .header("x-opencode-project", "global")
                .header("connection", "close")
                .body(Full::new(Bytes::new()))
                .context("building OpenCode Free model catalog request")?;
            match self.send(lane, &host, port, request).await {
                Ok(response) if response.status.is_success() => {
                    lane.mark_success();
                    return read_body_raw(response.body, MAX_CATALOG_BODY).await;
                }
                Ok(response) => {
                    let status = response.status;
                    let body = read_body_limited(response.body, MAX_ERROR_BODY).await;
                    let error = anyhow::anyhow!(
                        "OpenCode Free model catalog request failed\n  lane: {}\n  endpoint: {}/models\n  status: {}\n  response: {}",
                        lane.name,
                        api_base.trim_end_matches('/'),
                        status,
                        body
                    );
                    if retryable_status(status) {
                        lane.cool_down(self.cooldown);
                        jcode_base::logging::warn(&format!(
                            "OpenCode Free lane '{}' model catalog request failed with {status}; rotating",
                            lane.name
                        ));
                    }
                    last_error = Some(error);
                }
                Err(error) => {
                    lane.cool_down(self.cooldown);
                    jcode_base::logging::warn(&format!(
                        "OpenCode Free lane '{}' model catalog connection failed; rotating",
                        lane.name
                    ));
                    last_error = Some(error);
                }
            }
            excluded.push(index);
        }
        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!("OpenCode Free model catalog request failed after all lanes")
        }))
    }

    async fn stream(
        &self,
        response: FreeResponse,
        model: String,
        provider_pin: Arc<Mutex<Option<ProviderPin>>>,
        tx: mpsc::Sender<Result<StreamEvent>>,
    ) {
        let mut stream = OpenRouterStream::from_anyhow_stream(response.body, model, provider_pin);
        let idle = jcode_base::provider::stream_idle_timeout();
        loop {
            match tokio::time::timeout(idle, stream.next()).await {
                Ok(Some(Ok(event))) => {
                    if tx.send(Ok(event)).await.is_err() {
                        return;
                    }
                }
                Ok(Some(Err(error))) => {
                    let _ = tx
                        .send(Err(error.context("OpenCode Free stream error")))
                        .await;
                    return;
                }
                Ok(None) => return,
                Err(_) => {
                    let _ = tx
                        .send(Err(anyhow::anyhow!(
                            "OpenCode Free stream idle timeout after {}s",
                            idle.as_secs()
                        )))
                        .await;
                    return;
                }
            }
        }
    }
}

struct FreeResponse {
    status: http::StatusCode,
    headers: http::HeaderMap,
    body: std::pin::Pin<Box<dyn futures::Stream<Item = Result<Bytes>> + Send>>,
}

fn build_connector() -> Result<boring::ssl::SslConnector> {
    use boring::ssl::{SslConnector, SslMethod, SslVerifyMode};
    let mut builder = SslConnector::builder(SslMethod::tls())?;
    builder.set_verify(SslVerifyMode::PEER);
    builder.set_alpn_protos(b"\x08http/1.1")?;
    builder.set_grease_enabled(false);
    builder.enable_ocsp_stapling();
    builder.set_curves_list("X25519MLKEM768:X25519:P-256:P-384")?;
    builder.enable_signed_cert_timestamps();
    Ok(builder.build())
}

async fn connect(host: &str, port: u16, proxy: Option<&str>) -> Result<TcpStream> {
    let Some(proxy) = proxy else {
        return TcpStream::connect((host, port))
            .await
            .with_context(|| format!("connecting to {host}:{port}"));
    };

    let (proxy_host, proxy_port, proxy_authorization) = parse_proxy(proxy)?;
    let mut tcp = TcpStream::connect((proxy_host.as_str(), proxy_port))
        .await
        .with_context(|| format!("connecting to proxy {proxy_host}:{proxy_port}"))?;
    let _ = tcp.set_nodelay(true);
    let authorization = proxy_authorization
        .map(|value| format!("Proxy-Authorization: Basic {value}\r\n"))
        .unwrap_or_default();
    let request = format!(
        "CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\nProxy-Connection: Keep-Alive\r\n{authorization}\r\n"
    );
    tcp.write_all(request.as_bytes())
        .await
        .context("writing proxy CONNECT")?;

    let mut header = Vec::new();
    let mut byte = [0u8; 1];
    while !header.ends_with(b"\r\n\r\n") {
        if tcp.read(&mut byte).await? == 0 {
            anyhow::bail!("proxy closed before CONNECT response");
        }
        header.push(byte[0]);
        if header.len() > 8192 {
            anyhow::bail!("proxy CONNECT response too large");
        }
    }
    let response = String::from_utf8_lossy(&header);
    let status = response.lines().next().unwrap_or_default();
    if !status.contains(" 200") {
        anyhow::bail!("proxy CONNECT failed: {status}");
    }
    Ok(tcp)
}

fn parse_proxy(raw: &str) -> Result<(String, u16, Option<String>)> {
    let raw = raw.trim();
    let without_scheme = raw
        .strip_prefix("http://")
        .or_else(|| raw.strip_prefix("https://"))
        .unwrap_or(raw);
    let (credentials, authority) = without_scheme
        .rsplit_once('@')
        .map(|(credentials, authority)| (Some(credentials), authority))
        .unwrap_or((None, without_scheme));
    let authority = authority.trim_end_matches('/');
    let (host, port) = authority
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("proxy URL must include a port: {raw}"))?;
    let port = port
        .parse::<u16>()
        .with_context(|| format!("invalid proxy port in {raw}"))?;
    if host.is_empty() {
        anyhow::bail!("proxy URL has an empty host: {raw}");
    }
    let authorization = credentials
        .filter(|credentials| !credentials.is_empty())
        .map(|credentials| base64::engine::general_purpose::STANDARD.encode(credentials));
    Ok((
        host.trim_matches(['[', ']']).to_string(),
        port,
        authorization,
    ))
}

fn resolve_proxy(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("direct") || raw.eq_ignore_ascii_case("none") {
        return None;
    }
    if raw.eq_ignore_ascii_case("env") {
        return default_proxy();
    }
    Some(raw.to_string())
}

fn default_proxy() -> Option<String> {
    if no_proxy(API_HOST) {
        return None;
    }
    for key in [
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ] {
        if let Ok(value) = std::env::var(key) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn no_proxy(host: &str) -> bool {
    std::env::var("NO_PROXY")
        .or_else(|_| std::env::var("no_proxy"))
        .map(|value| {
            value.split(',').any(|entry| {
                let entry = entry.trim().trim_start_matches('.');
                !entry.is_empty() && host.ends_with(entry)
            })
        })
        .unwrap_or(false)
}

fn parse_api_endpoint(api_base: &str, endpoint: &str) -> Result<(String, u16, String)> {
    let api_base = api_base.trim().trim_end_matches('/');
    let Some(rest) = api_base.strip_prefix("https://") else {
        anyhow::bail!("OpenCode Free requires an https API base, got {api_base}");
    };
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = authority
        .rsplit_once(':')
        .filter(|(_, port)| port.parse::<u16>().is_ok())
        .map(|(host, port)| (host.to_string(), port.parse::<u16>().unwrap_or(443)))
        .unwrap_or_else(|| (authority.to_string(), 443));
    if host.is_empty() {
        anyhow::bail!("OpenCode Free API base has no host: {api_base}");
    }
    let path = if path.is_empty() {
        format!("/{endpoint}")
    } else {
        format!("/{}/{}", path.trim_matches('/'), endpoint)
    };
    Ok((host, port, path))
}

fn load_config() -> FileConfig {
    let Some(path) = config_path() else {
        return FileConfig::default();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return FileConfig::default();
    };
    match toml::from_str(&text) {
        Ok(config) => config,
        Err(error) => {
            jcode_base::logging::warn(&format!(
                "Ignoring invalid OpenCode Free config at {}: {error}",
                path.display()
            ));
            FileConfig::default()
        }
    }
}

fn config_path() -> Option<PathBuf> {
    jcode_base::storage::jcode_dir()
        .ok()
        .map(|dir| dir.join("opencode-free.toml"))
}

fn parse_strategy(value: &str) -> Option<Strategy> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "sticky" => Some(Strategy::Sticky),
        "round-robin" | "roundrobin" => Some(Strategy::RoundRobin),
        "lowest-usage" | "lowestusage" => Some(Strategy::LowestUsage),
        _ => None,
    }
}

fn retryable_status(status: http::StatusCode) -> bool {
    matches!(status.as_u16(), 401 | 403 | 408 | 425 | 429 | 500..=599)
}

fn parse_retry_after(headers: &http::HeaderMap) -> Option<Duration> {
    headers
        .get(http::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

async fn read_body_raw(
    mut body: std::pin::Pin<Box<dyn futures::Stream<Item = Result<Bytes>> + Send>>,
    limit: usize,
) -> Result<String> {
    let mut out = Vec::new();
    while let Some(chunk) = body.next().await {
        let bytes = chunk?;
        out.extend_from_slice(&bytes);
        if out.len() >= limit {
            break;
        }
    }
    out.truncate(limit);
    Ok(String::from_utf8_lossy(&out).into_owned())
}

async fn read_body_limited(
    mut body: std::pin::Pin<Box<dyn futures::Stream<Item = Result<Bytes>> + Send>>,
    limit: usize,
) -> String {
    let mut out = Vec::new();
    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(bytes) => {
                out.extend_from_slice(&bytes);
                if out.len() >= limit {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    out.truncate(limit);
    String::from_utf8_lossy(&out)
        .replace(['\n', '\r'], " ")
        .trim()
        .to_string()
}

/// The free lane requires an `<env>` block with a working directory inside
/// the system prompt. jcode's session context has the directory but not the
/// wrapper, so append one without disturbing the existing prompt.
pub fn ensure_env_block(system: &str) -> String {
    let cwd = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    if system.contains("<env>") && system.contains("Working directory:") {
        return system.to_string();
    }
    let mut out = String::with_capacity(system.len() + 256);
    out.push_str(system);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&format!(
        "<env>\n  Working directory: {cwd}\n  Workspace root folder: {cwd}\n  Platform: {}\n</env>\n",
        std::env::consts::OS
    ));
    out
}

/// Append whichever of the two gate tools the caller did not provide. Their
/// schemas must have non-empty `properties`; the gate inspects presence.
pub fn ensure_gate_tools(tools: &[ToolDefinition]) -> Vec<ToolDefinition> {
    let mut out = tools.to_vec();
    if !out.iter().any(|tool| tool.name == "bash") {
        out.push(ToolDefinition::new(
            "bash",
            "Run a shell command in the project workspace.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run." },
                    "description": { "type": "string", "description": "What the command does." }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        ));
    }
    if !out.iter().any(|tool| tool.name == "read") {
        out.push(ToolDefinition::new(
            "read",
            "Read a file from the filesystem.",
            json!({
                "type": "object",
                "properties": {
                    "filePath": { "type": "string", "description": "Absolute path of the file to read." },
                    "offset": { "type": "integer", "description": "Line number to start reading from." },
                    "limit": { "type": "integer", "description": "Maximum number of lines to read." }
                },
                "required": ["filePath"],
                "additionalProperties": false
            }),
        ));
    }
    out
}

/// Stable per-conversation signal. The first user turn is stable across turns
/// and provider forks, which is what keeps the upstream session id (and its
/// quota/cache affinity) from changing every request.
pub fn conversation_signal(messages: &[Message]) -> String {
    for message in messages {
        if message.role == Role::User
            && let Ok(signal) = serde_json::to_string(&message.content)
            && signal != "[]"
            && signal != "null"
        {
            return signal;
        }
    }
    String::new()
}

/// A fresh identity per process, so two instances with the same prompt are
/// different upstream users. `OPENCODE_FREE_SESSION_NONCE` pins it instead.
fn process_nonce() -> u64 {
    static NONCE: OnceLock<u64> = OnceLock::new();
    *NONCE.get_or_init(|| {
        let raw = std::env::var("OPENCODE_FREE_SESSION_NONCE").unwrap_or_else(|_| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            format!(
                "{}\0{}\0{now}",
                std::process::id(),
                std::thread::current().name().unwrap_or("")
            )
        });
        let digest = Sha256::digest(raw.as_bytes());
        u64::from_be_bytes(digest[..8].try_into().unwrap_or_default())
    })
}

fn mint_session_id(signal: &str) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let current = now.wrapping_mul(0x1000).wrapping_add(1) & 0xffff_ffff_ffff;
    let inverted = (!current) & 0xffff_ffff_ffff;
    static SESSION_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SESSION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let digest =
        Sha256::digest(format!("ses\0{}\0{sequence}\0{signal}", process_nonce()).as_bytes());
    let mut number = u128::from_be_bytes(digest[6..22].try_into().unwrap_or_default());
    let mut tail = [b'0'; 14];
    for slot in tail.iter_mut().rev() {
        *slot = BASE62[(number % 62) as usize];
        number /= 62;
    }
    format!(
        "ses_{inverted:012x}{}",
        std::str::from_utf8(&tail).unwrap_or("00000000000000")
    )
}

fn opencode_user_agent() -> String {
    if let Ok(raw) = std::env::var("OPENCODE_FREE_USER_AGENT") {
        let raw = raw.trim();
        if !raw.is_empty() {
            return raw.to_string();
        }
    }
    let platform = match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        "x86_64" => "x64",
        other => other,
    };
    format!("opencode/1.18.31 ({platform} {arch}; node22.18.0)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_matches_the_cli_chat_shape() {
        let mut body = json!({
            "model": "m",
            "max_tokens": 64,
            "provider": {"order": ["x"]},
            "stream_options": {"include_usage": false},
        });
        normalize_free_body(&mut body);
        assert_eq!(body["max_completion_tokens"], 64);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("provider").is_none());
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn session_ids_are_unique_per_mint() {
        assert_ne!(process_nonce(), 0);
        let first = mint_session_id("same prompt");
        let second = mint_session_id("same prompt");
        assert_ne!(first, second);
        assert_eq!(first.len(), 30);
        assert_eq!(second.len(), 30);
    }

    #[test]
    fn session_id_has_opencode_shape_and_varies_by_signal() {
        let id = mint_session_id("hello");
        assert_eq!(id.len(), 30);
        assert!(id.starts_with("ses_"));
        assert!(id[4..16].chars().all(|c| c.is_ascii_hexdigit()));
        assert!(id[16..].chars().all(|c| c.is_ascii_alphanumeric()));
        assert_ne!(id, mint_session_id("goodbye"));
    }

    #[test]
    fn session_is_stable_until_forced() {
        let mut store = SessionStore::new();
        let first = store.get_or_mint("conversation", false);
        assert_eq!(first, store.get_or_mint("conversation", false));
        assert_ne!(first, store.get_or_mint("conversation", true));
    }

    #[test]
    fn gate_tools_fill_missing_names() {
        let tools = ensure_gate_tools(&[]);
        assert!(tools.iter().any(|tool| tool.name == "read"));
        assert!(tools.iter().any(|tool| tool.name == "bash"));
        for tool in &tools {
            assert!(
                tool.input_schema
                    .get("properties")
                    .and_then(Value::as_object)
                    .is_some_and(|properties| !properties.is_empty())
            );
        }
    }

    #[test]
    fn env_block_wraps_working_directory() {
        let system = ensure_env_block("You are a test agent.");
        assert!(system.contains("<env>"));
        assert!(system.contains("Working directory:"));
        assert!(system.contains("</env>"));
    }

    #[test]
    fn proxy_parser_accepts_common_forms() {
        assert_eq!(
            parse_proxy("http://127.0.0.1:8080").unwrap(),
            ("127.0.0.1".to_string(), 8080, None)
        );
        let (host, port, authorization) = parse_proxy("user:pass@proxy.example:3128").unwrap();
        assert_eq!((host.as_str(), port), ("proxy.example", 3128));
        assert_eq!(authorization.as_deref(), Some("dXNlcjpwYXNz"));
    }
}
