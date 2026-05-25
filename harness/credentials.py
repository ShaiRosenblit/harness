"""Small credentials store for the harness.

Stores the OpenRouter API key at ``~/.config/harness/credentials.json`` with
0600 permissions. Plain JSON; not encrypted. Fine for a single-user dev box,
and the harness's threat model already assumes the disposable VPS is the
containment boundary.

If you want stronger storage (macOS Keychain, GNOME keyring, etc.) that's a
v2 swap — the public surface here is just ``load() / save() / inject_env()``.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Optional


CONFIG_DIR = Path(os.environ.get("HARNESS_CONFIG_DIR", str(Path.home() / ".config" / "harness")))
CREDENTIALS_PATH = CONFIG_DIR / "credentials.json"


def load() -> dict:
    """Return saved credentials dict, or {} if none."""
    if not CREDENTIALS_PATH.exists():
        return {}
    try:
        return json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def save(creds: dict) -> Path:
    """Write credentials dict atomically with 0600 perms."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CREDENTIALS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    os.replace(tmp, CREDENTIALS_PATH)
    return CREDENTIALS_PATH


def get_api_key() -> Optional[str]:
    """Return the saved OpenRouter API key (or None). Env var wins if set."""
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env
    return load().get("openrouter_api_key")


def save_api_key(key: str) -> Path:
    creds = load()
    creds["openrouter_api_key"] = key.strip()
    return save(creds)


def clear_api_key() -> None:
    creds = load()
    creds.pop("openrouter_api_key", None)
    save(creds)


# ---- Telegram bridge --------------------------------------------------- #
# Stored in the same credentials.json so /telegram login persists across
# UI restarts, exactly like /login does for the OpenRouter key. Env vars
# (TELEGRAM_BOT_TOKEN / TELEGRAM_ALLOWED_CHAT_IDS) still win if set, so
# a one-off shell export overrides the saved value.


def get_telegram_token() -> Optional[str]:
    env = os.environ.get("TELEGRAM_BOT_TOKEN")
    if env:
        return env.strip()
    return load().get("telegram_bot_token")


def save_telegram_token(token: str) -> Path:
    creds = load()
    creds["telegram_bot_token"] = token.strip()
    return save(creds)


def clear_telegram_token() -> None:
    creds = load()
    creds.pop("telegram_bot_token", None)
    save(creds)


def get_telegram_allowed_ids() -> list[int]:
    """Return allowed numeric Telegram chat/user ids. Env wins if set."""
    raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS")
    if raw is None:
        raw_list = load().get("telegram_allowed_chat_ids") or []
        # Stored as a list of ints already; tolerate strings too.
        return [int(x) for x in raw_list]
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def save_telegram_allowed_ids(ids: list[int]) -> Path:
    creds = load()
    creds["telegram_allowed_chat_ids"] = [int(x) for x in ids]
    return save(creds)


def clear_telegram_allowed_ids() -> None:
    creds = load()
    creds.pop("telegram_allowed_chat_ids", None)
    save(creds)


def inject_env() -> bool:
    """If a key is saved (and the env var isn't already set), inject it into
    os.environ so harness.model._client() can pick it up. Returns True if a
    key is now present in the env."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return True
    key = load().get("openrouter_api_key")
    if not key:
        return False
    os.environ["OPENROUTER_API_KEY"] = key
    return True


def mask(key: str, show: int = 4) -> str:
    """Render a key for display: 'sk-or-xxxx...abcd'."""
    if not key:
        return ""
    if len(key) <= show * 2:
        return "*" * len(key)
    return key[:show] + "..." + key[-show:]
