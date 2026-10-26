# OpenCode Free

`opencode-free` talks to the anonymous OpenCode Zen free tier at
`https://opencode.ai/zen/v1` with `Bearer public`. No account, signup, or API
key is required.

```bash
jcode -p opencode-free
jcode -p opencode-free -m deepseek-v4.1-flash-free
jcode model list -p opencode-free   # live list, filtered to -free ids
jcode run -p opencode-free "summarize this repository"
```

Aliases: `oc-free`, `opencode-anonymous`, `free`.

The default model is `deepseek-v4.1-flash-free`. Model ids are discovered from
the live catalog, so new free models appear without a jcode update.

Requests reproduce the CLI's wire shape: the `opencode/...` User-Agent, the
session/affinity headers, the `<env>` block, `bash` + `read` tools, `stream:
true`, `store: false`, and `max_completion_tokens`. That shape is what the free
gate checks; a plain OpenAI request is refused no matter which key it carries.

## Lanes, proxies, and rotation

A *lane* is one egress path: a proxy URL, optionally with a real API key. With
more than one lane, jcode rotates between them when a request is refused.

```toml
# ~/.jcode/opencode-free.toml
strategy = "lowest-usage"   # sticky | round-robin | lowest-usage
max_attempts = 3
cooldown_secs = 60

[[accounts]]
name = "home"
proxy = "http://127.0.0.1:8080"

[[accounts]]
name = "work"
proxy = "http://user:pass@10.0.0.5:3128"
api_key = "public"

[[accounts]]
name = "direct"
# no proxy key = direct connection
```

`lowest-usage` (default) prefers the lane with the fewest requests and breaks a
tie by the least recently used lane, `sticky` keeps using the first healthy lane
(the prompt-cache-friendly choice), and `round-robin` cycles through them. A
lane that answers `401/403/408/425/429/5xx` cools down with exponential backoff
(`cooldown_secs`, doubling per consecutive failure, capped at 300s) and the next
lane is tried up to `max_attempts` times. A success clears that lane's failure
count.

Environment overrides, in precedence order:

| variable | meaning |
| --- | --- |
| `OPENCODE_FREE_LANES` | Comma-separated proxy URLs replacing all config lanes. Each item may be `none` or `direct` for a direct connection, e.g. `http://a:8080,http://b:8080,none`. |
| `OPENCODE_FREE_PROXY` | One-lane replacement for the config lanes. `none`/`direct` means no proxy, `env` means read `HTTPS_PROXY`/`HTTP_PROXY`, a URL means that proxy. |
| `OPENCODE_FREE_API_KEY` | Bearer key for lanes that do not set one (`public` is the free default). |
| `OPENCODE_FREE_STRATEGY` | `sticky`, `round-robin`, or `lowest-usage`. |
| `OPENCODE_FREE_MAX_ATTEMPTS` | Retry count across lanes, clamped to 1-10. |
| `OPENCODE_FREE_COOLDOWN_SECS` | Lane cooldown after a retryable refusal. |
| `OPENCODE_FREE_USER_AGENT` | Overrides the `opencode/...` User-Agent; keep the CLI shape if you change it. |
| `OPENCODE_FREE_SESSION_NONCE` | Pins the session identity; leave unset for a fresh identity per process. |

When no lane variable is set, the config file is used. When neither exists, one
lane uses the process's `HTTPS_PROXY` (honouring `NO_PROXY`) or a direct
connection.

## Multiple instances

Each process owns its lane cursor, cooldowns, and session store. Run as many
instances as you want, each with its own egress:

```bash
# instance A: proxy 1
OPENCODE_FREE_PROXY=http://127.0.0.1:8081 jcode -p opencode-free

# instance B: proxy 2
OPENCODE_FREE_PROXY=http://127.0.0.1:8082 jcode -p opencode-free

# instance C: no proxy at all
OPENCODE_FREE_PROXY=none jcode -p opencode-free

# instance D: rotate over three proxies
OPENCODE_FREE_LANES='http://a:8080,http://b:8080,none' \
  OPENCODE_FREE_STRATEGY=round-robin jcode -p opencode-free
```

The same process can use the shared config instead:

```bash
jcode -p opencode-free    # uses ~/.jcode/opencode-free.toml lanes
```

Every new process mints a new `ses_...` identity from a pid + start-time nonce,
so two instances sending the same first prompt are separate upstream users,
each with separate quota accounting. Within one process the id stays stable for
the conversation so prefix caches still hit; a gate rejection re-mints a fresh
id (the 12 hex digits are the CLI's inverted timestamp, the 14 Base62 digits
carry the conversation identity). Lane health and the rotation cursor are
in-process too, so one instance's cooldown never stalls another.

## Unattended runs

`jcode run` is already non-interactive (tools execute without a prompt). To keep
the model from ending a turn on a question, enable autonomous answers:

```bash
jcode -p opencode-free --auto-answer-first
```

or persistently in `~/.jcode/config.toml`:

```toml
[features]
auto_answer_first = true
```

or for one shell:

```bash
JCODE_AUTO_ANSWER_FIRST=1 jcode -p opencode-free
```

This appends an `# Autonomous Mode` block to the system prompt: when a decision
is needed, the model takes the first reasonable option it would have offered,
states it in one line, and continues. It does not bypass tool permissions or
destructive-action safeguards. Combine with `jcode run` for a fully headless
invocation.

## Queuing messages in the TUI

Queueing is built in; nothing needs to be enabled.

- While a turn is running, type the next prompt. In the default immediate mode,
  **Enter** interleaves it and **Ctrl+Enter** (Cmd+Enter on macOS) queues it
  until the current turn finishes.
- **Ctrl+T** toggles queue mode: **Enter** queues, **Ctrl+Enter** sends now.
  The status line confirms the mode.
- Queued prompts are sent in order as soon as the turn completes; you can queue
  several while the agent works.
- `/clear` clears the conversation, the queue, and the display.

`jcode run` takes one prompt and exits; use the TUI when you want to queue
follow-ups.
