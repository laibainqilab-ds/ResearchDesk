"""Repository-layer tests against a real (in-memory SQLite) ORM session --
see tests/conftest.py's db_session fixture. These exercise actual SQL
(joins, uniqueness, cascades), not mocks, since that's the behavior this
layer exists to get right.
"""

from app.repositories.chat_repository import ChatRepository
from app.repositories.context_link_repository import ContextLinkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.run_repository import RunRepository
from app.repositories.user_repository import UserRepository


def make_user(db_session, email="user@example.com"):
    return UserRepository(db_session).create(email=email, password_hash="hashed")


# ---------------------------------------------------------------------------
# UserRepository
# ---------------------------------------------------------------------------

def test_user_repository_create_and_get_by_email(db_session):
    user = make_user(db_session)

    assert UserRepository(db_session).get_by_email("user@example.com").id == user.id


def test_user_repository_get_by_id_missing_returns_none(db_session):
    assert UserRepository(db_session).get_by_id("does-not-exist") is None


# ---------------------------------------------------------------------------
# ChatRepository
# ---------------------------------------------------------------------------

def test_chat_repository_list_for_user_only_returns_owned_chats(db_session):
    owner = make_user(db_session, "owner@example.com")
    other = make_user(db_session, "other@example.com")
    chats = ChatRepository(db_session)

    mine = chats.create(owner.id, title="Mine")
    chats.create(other.id, title="Theirs")

    result = chats.list_for_user(owner.id)

    assert [chat.id for chat in result] == [mine.id]


def test_chat_repository_get_for_user_returns_none_for_non_owner(db_session):
    owner = make_user(db_session, "owner@example.com")
    other = make_user(db_session, "other@example.com")
    chats = ChatRepository(db_session)
    chat = chats.create(owner.id)

    assert chats.get_for_user(chat.id, other.id) is None
    assert chats.get_for_user(chat.id, owner.id) is not None


def test_chat_repository_delete_cascades_to_messages_and_links(db_session):
    owner = make_user(db_session)
    chats = ChatRepository(db_session)
    messages = MessageRepository(db_session)
    links = ContextLinkRepository(db_session)

    source_chat = chats.create(owner.id, title="Source")
    target_chat = chats.create(owner.id, title="Target")
    messages.add_user_message(source_chat.id, "hello")
    links.create(owner.id, source_chat.id, target_chat.id)

    chats.delete(target_chat)

    assert chats.get(target_chat.id) is None
    assert links.list_for_source_chat(source_chat.id) == []
    assert messages.list_for_chat(source_chat.id) != []  # unrelated chat untouched


# ---------------------------------------------------------------------------
# MessageRepository
# ---------------------------------------------------------------------------

def test_message_repository_add_assistant_message_persists_sources(db_session):
    owner = make_user(db_session)
    chat = ChatRepository(db_session).create(owner.id)
    messages = MessageRepository(db_session)

    message = messages.add_assistant_message(
        chat_id=chat.id,
        content="Here is the answer [1].",
        mode="rag",
        sources=[
            {"citation_id": 1, "document_id": "doc1", "filename": "a.pdf", "page_number": 2, "chunk_id": 3, "rerank_score": 0.9},
        ],
    )

    assert len(message.sources) == 1
    assert message.sources[0].filename == "a.pdf"
    assert message.sources[0].rerank_score == 0.9


def test_message_repository_list_for_chat_orders_chronologically(db_session):
    owner = make_user(db_session)
    chat = ChatRepository(db_session).create(owner.id)
    messages = MessageRepository(db_session)

    messages.add_user_message(chat.id, "first")
    messages.add_assistant_message(chat.id, "second", mode="rag")

    result = messages.list_for_chat(chat.id)

    assert [message.content for message in result] == ["first", "second"]


# ---------------------------------------------------------------------------
# DocumentRepository
# ---------------------------------------------------------------------------

def test_document_repository_upsert_then_list_for_owner(db_session):
    owner = make_user(db_session)
    documents = DocumentRepository(db_session)

    documents.upsert_indexed(
        document_id="doc1", owner_id=owner.id, filename="a.pdf",
        file_type="pdf", page_count=3, chunk_count=5,
    )

    result = documents.list_for_owner(owner.id)

    assert len(result) == 1
    assert result[0].chunk_count == 5


def test_document_repository_mark_deleted_excludes_from_listing(db_session):
    owner = make_user(db_session)
    documents = DocumentRepository(db_session)
    document = documents.upsert_indexed(
        document_id="doc1", owner_id=owner.id, filename="a.pdf",
        file_type="pdf", page_count=None, chunk_count=2,
    )

    documents.mark_deleted(document)

    assert documents.list_for_owner(owner.id) == []


# ---------------------------------------------------------------------------
# RunRepository
# ---------------------------------------------------------------------------

def test_run_repository_create_and_get_for_message(db_session):
    owner = make_user(db_session)
    chat = ChatRepository(db_session).create(owner.id)
    message = MessageRepository(db_session).add_assistant_message(chat.id, "answer", mode="rag")
    runs = RunRepository(db_session)

    run = runs.create(
        chat_id=chat.id, message_id=message.id, mode="rag",
        retrieval={"original_question": "q"},
    )

    assert runs.get_for_message(message.id).id == run.id
    assert runs.list_for_chat(chat.id) == [run]


# ---------------------------------------------------------------------------
# ContextLinkRepository
# ---------------------------------------------------------------------------

def test_context_link_repository_create_and_list(db_session):
    owner = make_user(db_session)
    chats = ChatRepository(db_session)
    links = ContextLinkRepository(db_session)
    source = chats.create(owner.id)
    target = chats.create(owner.id)

    links.create(owner.id, source.id, target.id)

    result = links.list_for_source_chat(source.id)
    assert len(result) == 1
    assert result[0].target_chat_id == target.id
