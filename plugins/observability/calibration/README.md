# Calibration Plugin for Hermes

Inspired by [gbrain](https://github.com/garrytan/gbrain)'s calibration system (v0.36+).

## What it does

Tracks tool-call outcomes per domain, detects recurring failure patterns, and warns the agent before repeating known mistakes.

## How it works

1. **Records** every tool call's outcome (success/failure) with inferred domain
2. **Aggregates** results per domain (e.g. `azure_deploy`, `powershell`, `csharp`)
3. **Detects** bias patterns when accuracy drops below 50% over ≥5 samples
4. **Nudges** the agent with a warning when calling a tool in a biased domain
5. **Cooldowns** prevent spam — same pattern nudges at most once per 14 days

## Schema

SQLite database at `~/.hermes/calibration.db`:

- `judgments` — prediction + domain + confidence
- `outcomes` — actual result (success/failure/partial)
- `bias_patterns` — detected recurring biases with accuracy rates
- `nudge_log` — cooldown tracking per pattern

## Domain inference

Automatically inferred from tool name + arguments:

| Tool pattern | Argument hint | Domain |
|-------------|---------------|--------|
| `node_invoke`, `node_lib` | — | `node_operations` |
| `terminal` | `powershell` | `powershell` |
| `terminal` | `dotnet` | `csharp` |
| `browser_*`, `web_*` | — | `web_research` |
| `patch`/`write_file` | `.bicep` | `azure_deploy` |
| `patch`/`write_file` | `.py` | `python` |
| `execute_code` | — | `python` |
| `read_file`, `search_files` | — | `file_ops` |
| `terminal` | `git` | `git` |
| `cronjob` | — | `scheduling` |
| `hindsight_*`, `memory` | — | `memory` |

## Configuration

No configuration required. The plugin is automatically discovered when placed in `plugins/observability/calibration/`.

Enable via:
```bash
hermes plugins enable observability/calibration
```

## Tuning parameters

Edit the constants in `__init__.py`:

- `MIN_BUCKET_N = 5` — minimum samples before declaring bias
- `ACCURACY_THRESHOLD = 0.5` — below this = bias detected
- `NUDGE_COOLDOWN_DAYS = 14` — cooldown between nudges for same pattern

## Reset

To clear all calibration data:
```bash
rm ~/.hermes/calibration.db
```
