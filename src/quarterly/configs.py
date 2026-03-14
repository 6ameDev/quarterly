import json
import os
from pathlib import Path

from dotenv import load_dotenv

# --- Environment Configuration (Service) ---

# Load development environment variables by default
load_dotenv(".env.development")


def get_base_url() -> str:
    return os.getenv("BASE_URL", "http://localhost:11434")


def get_embed_model_name() -> str:
    return os.getenv("EMBED_MODEL_NAME", "nomic-embed-text:latest")


def get_llm_model_name() -> str:
    return os.getenv("LLM_MODEL_NAME", "qwen2.5:1.5b")


def get_system_prompt() -> str:
    raw_prompt = os.getenv("SYSTEM_PROMPT", "")
    return raw_prompt.replace("\\n", "\n")


def get_persist_dir() -> str:
    return os.getenv("PERSIST_DIR", "./chroma_db")


def get_collection_name() -> str:
    return os.getenv("COLLECTION_NAME", "quarterly_docs_nomic")


# --- User Configuration (CLI) ---

USER_CONFIG_FILE = Path.home() / ".quarterly.json"
DEFAULT_USER_HOST = "http://localhost:8000"


def load_user_config() -> dict:
    """Load user-specific configurations from a local JSON file."""
    if USER_CONFIG_FILE.exists():
        try:
            with open(USER_CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            return {"host": DEFAULT_USER_HOST}
    return {"host": DEFAULT_USER_HOST}


def save_user_config(config: dict) -> None:
    """Save user-specific configurations to a local JSON file."""
    try:
        with open(USER_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Failed to save user config: {e}")


def get_user_host() -> str:
    """Retrieve the saved server host from user config."""
    config = load_user_config()
    return config.get("host", DEFAULT_USER_HOST)


def update_user_host(new_host: str) -> None:
    """Update and persist the server host in user config."""
    config = load_user_config()
    config["host"] = new_host.strip().rstrip("/")
    save_user_config(config)
