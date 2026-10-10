import os
import secrets

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7


def _generate_and_persist_jwt_secret() -> str:
    """Dev convenience: a missing JWT_SECRET is generated once and appended
    to .env so tokens stay valid across process restarts, instead of
    silently rotating (and invalidating every session) on every reload."""
    from pathlib import Path

    secret = secrets.token_urlsafe(48)
    env_path = Path(".env")

    try:
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        lines = [line for line in existing.splitlines() if not line.startswith("JWT_SECRET=")]
        lines.append(f"JWT_SECRET={secret}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass

    return secret


if not JWT_SECRET:
    JWT_SECRET = _generate_and_persist_jwt_secret()
