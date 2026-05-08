# API Server 再設計

## 現状の問題

`gateway/platforms/api_server.py` は 4144 行のモノリシックファイル。責務が混在し、以下の課題がある。

1. **Gateway への強い依存** — `_create_agent()` が `gateway.run` のプライベート関数 (`_resolve_runtime_agent_kwargs`, `_resolve_gateway_model`, `_load_gateway_config`, `GatewayRunner`) に直接依存。API Server を Gateway なしで起動できない。
2. **SSE 送信の重複** — `_write_sse_chat_completion` (1944 行~) と `_write_sse_responses` (2077 行~) で CORS ヘッダ構築、キープアライブ、接続切断ハンドリング、drain 処理が重複。
3. **コールバックシグネチャの非統一** — `tool_progress_callback` が `_make_run_event_callback` 内では `(event_type, tool_name, preview, args, **kwargs)` だが、SSE 書き込み側では別の形式で解釈。型安全がない。
4. **CLI の未独立** — `hermes gateway` 経由のプラットフォーム起動のみ。`hermes api-server run` といった独立 CLI が存在しない。
5. **設定の混在** — Gateway 用 config.yaml を API Server も読み込んでいるが、API Server 固有の設定分離が不十分。

## 設計方針

後方互換を保ちつつ、段階的に責務を分離する。

- **Phase 1: パッケージ化** — `gateway/platforms/api_server.py` → `api_server/` パッケージへ移行
- **Phase 2: Agent Factory 独立** — Gateway 依存を排除し、設定注入型に
- **Phase 3: SSE 一元化** — 共通送信レイヤーを分離
- **Phase 4: イベントバス標準化** — dataclass ベースの型安全イベント
- **Phase 5: 独立 CLI** — `hermes api-server run` サブコマンド
- **Phase 6: 統合テスト** — 各レイヤーのテスト

## ディレクトリ構成

```
api_server/
├── __init__.py              # 公開 API（後方互換用 Adapter もここへ）
├── server.py                # aiohttp Application / ルーティング / ミドルウェア
├── config.py                # API Server 固有設定 + 既存 config.yaml 読み込みラッパー
├── agent_factory.py         # AIAgent 生成（Gateway 非依存）
├── sse.py                   # SSE ストリーム書き込み・drain・キープアライブ一元管理
├── events.py                # StreamEvent dataclass + コールバック統一インターフェース
├── handlers/                # エンドポイントハンドラ群
│   ├── __init__.py
│   ├── chat_completions.py
│   ├── responses.py
│   ├── runs.py
│   ├── sessions.py
│   ├── memory.py
│   ├── skills.py
│   ├── config.py
│   ├── jobs.py
│   ├── nodes.py
│   └── health.py
├── store.py                 # ResponseStore / RunStatusStore（軽量永続化）
└── cli.py                   # `hermes api-server run` サブコマンド
```

## モジュール責務

### `api_server/config.py`

- `API_SERVER_HOST`, `API_SERVER_PORT`, `API_SERVER_KEY` などの環境変数読み取り
- `~/.hermes/config.yaml` の `api_server:` セクション読み取り
- `load_api_server_config() -> APIServerConfig`（dataclass）を提供
- Gateway の `_load_gateway_config` / `_resolve_gateway_model` と同等の機能を、Gateway モジュールに依存せず実装

### `api_server/agent_factory.py`

- `create_agent(config: APIServerConfig, *, callbacks: AgentCallbacks, ...) -> AIAgent`
- 必要な設定値は `APIServerConfig` から取得。`gateway.run` への import を排除
- `AgentCallbacks` dataclass でコールバックを型付けして注入

### `api_server/sse.py`

- `SSEWriter` クラス — `StreamResponse` ラッパー
  - `prepare()` — CORS + 共通ヘッダ設定
  - `write_event(event_type, data)` — イベント書き込み
  - `write_delta(content)` — Chat Completions 用 delta 書き込み
  - `keepalive()` — 自動キープアライブ（タイマー制御）
  - `drain()` — 接続切断時のクリーンアップ
  - `close()` — 終了マーカー書き込み + drain
- 各ハンドラは `SSEWriter` を使うだけで、drain/keepalive/切断処理を個別に書かない

### `api_server/events.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class StreamEvent:
    event: str
    timestamp: float
    data: dict

@dataclass
class ToolStartedEvent(StreamEvent):
    tool: str
    preview: str
    args: Optional[dict] = None

@dataclass
class ToolCompletedEvent(StreamEvent):
    tool: str
    duration: float
    is_error: bool

@dataclass
class TextDeltaEvent(StreamEvent):
    delta: str

class EventBus:
    """型安全なコールバック登録・配信"""
    def on_tool_started(self, callback: Callable[[ToolStartedEvent], None]): ...
    def on_tool_completed(self, callback: Callable[[ToolCompletedEvent], None]): ...
    def on_text_delta(self, callback: Callable[[TextDeltaEvent], None]): ...
    def emit(self, event: StreamEvent): ...
```

AIAgent 側の `tool_progress_callback` シグネチャをこの `EventBus` に統合し、型安全にする。

### `api_server/server.py`

- aiohttp `Application` 構築
- ミドルウェア登録（CORS、body limit、security headers）
- ルーティング登録（各ハンドラを `handlers/` から import）
- 起動・停止制御

### `api_server/cli.py`

- `hermes api-server run` サブコマンド
- `hermes api-server status`
- `hermes api-server stop`
- SIGTERM/SIGINT ハンドリング
- systemd 対応（foreground + `--daemon` フラグ）

## Agent 初期化の Gateway 依存排除

現状: `_create_agent()` 内で以下を import
- `gateway.run._resolve_runtime_agent_kwargs`
- `gateway.run._resolve_gateway_model`
- `gateway.run._load_gateway_config`
- `gateway.run.GatewayRunner._load_reasoning_config`
- `gateway.run.GatewayRunner._load_fallback_model`

設計: `api_server/config.py` で同等の解決を行い、`APIServerConfig` に集約。

```python
# api_server/config.py
@dataclass
class APIServerConfig:
    host: str
    port: int
    api_key: Optional[str]
    cors_origins: tuple[str, ...]
    model_name: str
    provider: str
    api_key_provider: str
    base_url: Optional[str]
    api_mode: str
    reasoning_config: Optional[dict]
    fallback_model: Optional[str]
    enabled_toolsets: list[str]
    max_iterations: int
```

`load_api_server_config()` は内部で `hermes_cli.config.read_raw_config()` や `hermes_cli.runtime_provider.resolve_runtime_provider()` を呼ぶ。これらは `gateway/` ではなく `hermes_cli/` 下のモジュールなので、API Server として独立して import 可能。

## 移行計画

1. **Step 1** — `api_server/` パッケージを新設し、`__init__.py` で現行 `APIServerAdapter` をエクスポート（後方互換）
2. **Step 2** — `config.py` / `agent_factory.py` を実装。`gateway.run` 依存を移行
3. **Step 3** — `sse.py` を実装。`_write_sse_chat_completion` / `_write_sse_responses` の共通部分を抽出
4. **Step 4** — `events.py` を実装。`EventBus` + dataclass イベントを導入
5. **Step 5** — `handlers/` を切り出し。大きいハンドラから順に移行
6. **Step 6** — `cli.py` を実装。`hermes api-server run` を追加
7. **Step 7** — `gateway/platforms/api_server.py` を薄いラッパーに（`from api_server import APIServerAdapter`）
8. **Step 8** — 統合テスト

## 後方互換

- `gateway/platforms/api_server.py` はしばらくの間 `from api_server import APIServerAdapter` する薄いラッパーとして残す
- Gateway 経由の起動（`API_SERVER_ENABLED=true` + `hermes gateway`）も引き続き動作
- 環境変数・config.yaml の設定キーは変更しない
