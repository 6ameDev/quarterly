import asyncio
import contextlib
from pathlib import Path

import aiofiles
import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

from quarterly import configs

style = Style.from_dict(
    {
        "app": "#FFFFBA bold",
    }
)


async def check_server_health(host: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{host}/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


async def handle_host(new_host_str: str | None, current_host: str) -> str:
    if not new_host_str:
        print(f"Current host: {current_host}")
        return current_host

    new_host = new_host_str.strip().rstrip("/")
    if not new_host:
        print(f"Current host: {current_host}")
        return current_host

    configs.update_user_host(new_host)
    print(f"Host updated to: {new_host}")

    if await check_server_health(new_host):
        print(f"{new_host} is reachable.")
    else:
        print(f"Warning: {new_host} is not reachable.")

    return new_host


async def handle_ask(host: str, question: str):
    try:
        async with (
            httpx.AsyncClient() as client,
            client.stream("POST", f"{host}/ask", json={"question": question}, timeout=120.0) as response,
        ):
            if response.status_code != 200:
                text = await response.aread()
                print(f"\nError: Server returned status {response.status_code}: {text.decode('utf-8')}")
                return

            async for chunk in response.aiter_text():
                print(chunk, end="", flush=True)
            print("\n")
    except httpx.ConnectError:
        print(f"\nError: Could not connect to {host}. Is the server running?")
    except Exception as e:
        print(f"\nError: An error occurred: {e}")


async def ingest_file(host: str, filepath: Path, client: httpx.AsyncClient):
    try:
        async with aiofiles.open(filepath, encoding="utf-8") as f:
            content = await f.read()

        payload = {"text": content, "metadata": {"filename": filepath.name}}

        response = await client.post(f"{host}/ingest", json=payload, timeout=60.0)
        if response.status_code == 200:
            print(f"Ingested {filepath.name}")
        else:
            print(f"Failed to ingest {filepath.name}: {response.text}")
    except UnicodeDecodeError:
        print(f"Skipped {filepath.name} (not a valid UTF-8 text file)")
    except Exception as e:
        print(f"Error ingesting {filepath.name}: {e}")


async def handle_ingest(host: str, path_str: str):
    path = Path(path_str).expanduser()
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return

    if not await check_server_health(host):
        print(f"Error: Could not connect to server at {host}")
        return

    async with httpx.AsyncClient() as client:
        if path.is_file():
            await ingest_file(host, path, client)
        elif path.is_dir():
            print(f"Scanning directory: {path}")
            files = [f for f in path.rglob("*") if f.is_file() and not f.name.startswith(".")]
            if not files:
                print("No eligible files found in directory.")
                return

            print(f"Found {len(files)} files. Starting batch ingestion...")
            for f in files:
                await ingest_file(host, f, client)
        else:
            print(f"Error: Path is neither a file nor a directory: {path}")


def print_help():
    print("""
Available Commands:
  /ask <question>   - Ask a question based on ingested documents
  /ingest <path>    - Ingest a file or an entire directory of text files
  /host             - Show current server host URL
  /host <url>       - Set the server host URL (e.g. http://127.0.0.1:8000)
  /help             - Show this help message
  /exit or /quit    - Exit the application
""")


async def repl():
    host = configs.get_user_host()

    print(f"Quarterly CLI initialized. Target server: {host}")

    if await check_server_health(host):
        print("Server is reachable and healthy.")
    else:
        print("Warning: Server is not reachable. Some commands may fail.")

    print("Type /help to see available commands.\n")

    session = PromptSession()

    while True:
        try:
            text = await session.prompt_async([("class:app", "quarterly> ")], style=style)
            text = text.strip()

            if not text:
                continue

            if text in ("/exit", "/quit"):
                break

            elif text == "/help":
                print_help()

            elif text.startswith("/host"):
                parts = text.split(" ", 1)
                new_host_arg = parts[1].strip() if len(parts) > 1 else None
                host = await handle_host(new_host_arg, host)

            elif text.startswith("/ingest"):
                parts = text.split(" ", 1)
                if len(parts) < 2 or not parts[1].strip():
                    print("Usage: /ingest <path/to/file_or_dir>")
                else:
                    await handle_ingest(host, parts[1].strip())

            elif text.startswith("/ask"):
                parts = text.split(" ", 1)
                if len(parts) < 2 or not parts[1].strip():
                    print("Usage: /ask <your question here>")
                else:
                    await handle_ask(host, parts[1].strip())

            else:
                print("Error: Unknown command or malformed input.\nType /help to see available commands.")

        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def run():
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(repl())
    print("Goodbye!")

if __name__ == "__main__":
    run()

