.PHONY: install run cli test lint format clean

# Install all dependencies (including dev)
install:
	uv sync --all-groups

# Run the server
server:
	uv run api

# Run the REPL
repl:
	uv run cli

# Run tests
test:
	uv run pytest -v

# Run linter
lint:
	uv run ruff check .

# Format code
format:
	uv run ruff format .
	uv run ruff check --fix .

# Remove caches and build artifacts
clean:
	rm -rf .venv dist .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
