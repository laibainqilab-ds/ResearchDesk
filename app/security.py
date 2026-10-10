"""Password hashing and bearer-token issuance/verification for ResearchDesk's
minimal real-auth layer: hashed passwords, a signed JWT bearer token, no
hardcoded or fake users.

Uses the `bcrypt` package directly rather than passlib: passlib 1.7.4 (the
latest release) is unmaintained and its bcrypt backend self-test crashes
against bcrypt>=4.1's stricter 72-byte-password error handling.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALGORITHM, JWT_SECRET

# bcrypt only uses the first 72 bytes of its input -- truncate explicitly
# rather than letting long passwords raise or silently differ by backend.
_MAX_PASSWORD_BYTES = 72


def _encode(plain_password: str) -> bytes:
    return plain_password.encode("utf-8")[:_MAX_PASSWORD_BYTES]


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(_encode(plain_password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(_encode(plain_password), password_hash.encode("utf-8"))


def create_access_token(user_id: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "exp": expires_at}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


class InvalidTokenError(Exception):
    """Raised when a bearer token is missing, malformed, expired, or
    otherwise fails verification."""


def decode_access_token(token: str) -> str:
    """Return the user_id embedded in a valid access token, or raise
    InvalidTokenError."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as error:
        raise InvalidTokenError(str(error)) from error

    user_id = payload.get("sub")

    if not user_id:
        raise InvalidTokenError("Token payload missing 'sub'.")

    return user_id
