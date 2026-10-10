import time

import jwt
import pytest

from app.config import JWT_ALGORITHM, JWT_SECRET
from app.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_does_not_store_plaintext():
    password_hash = hash_password("correct horse battery staple")

    assert password_hash != "correct horse battery staple"


def test_verify_password_accepts_correct_password():
    password_hash = hash_password("correct horse battery staple")

    assert verify_password("correct horse battery staple", password_hash) is True


def test_verify_password_rejects_incorrect_password():
    password_hash = hash_password("correct horse battery staple")

    assert verify_password("wrong password", password_hash) is False


def test_hash_password_handles_passwords_longer_than_72_bytes():
    long_password = "x" * 200

    password_hash = hash_password(long_password)

    assert verify_password(long_password, password_hash) is True


def test_create_and_decode_access_token_roundtrip():
    token = create_access_token("user-123")

    assert decode_access_token(token) == "user-123"


def test_decode_access_token_rejects_garbage_token():
    with pytest.raises(InvalidTokenError):
        decode_access_token("not-a-real-token")


def test_decode_access_token_rejects_expired_token():
    expired_payload = {"sub": "user-123", "exp": int(time.time()) - 10}
    expired_token = jwt.encode(expired_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    with pytest.raises(InvalidTokenError):
        decode_access_token(expired_token)


def test_decode_access_token_rejects_token_signed_with_wrong_secret():
    forged_token = jwt.encode({"sub": "user-123", "exp": int(time.time()) + 60}, "wrong-secret", algorithm=JWT_ALGORITHM)

    with pytest.raises(InvalidTokenError):
        decode_access_token(forged_token)
