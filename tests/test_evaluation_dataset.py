import json
from pathlib import Path

import pytest

DATASET_PATH = Path("evaluation/rag_evaluation.json")
VALID_CATEGORIES = {
    "direct",
    "multi_chunk",
    "multi_document",
    "multi_hop",
    "ambiguous",
    "follow_up",
    "unanswerable",
}
REQUIRED_KEYS = {
    "id", "question", "category", "expected_answer", "key_facts",
    "should_abstain", "supporting_sources", "depends_on",
}


@pytest.fixture(scope="module")
def dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_dataset_has_at_least_50_questions(dataset):
    assert len(dataset) >= 50


def test_dataset_ids_are_unique(dataset):
    ids = [question["id"] for question in dataset]
    assert len(ids) == len(set(ids))


def test_every_question_has_required_keys(dataset):
    for question in dataset:
        assert REQUIRED_KEYS.issubset(question.keys()), question["id"]


def test_every_category_is_valid(dataset):
    for question in dataset:
        assert question["category"] in VALID_CATEGORIES, question["id"]


def test_all_required_categories_are_represented(dataset):
    categories_present = {question["category"] for question in dataset}
    assert VALID_CATEGORIES.issubset(categories_present)


def test_unanswerable_questions_have_no_ground_truth(dataset):
    for question in dataset:
        if question["should_abstain"]:
            assert question["supporting_sources"] == [], question["id"]
            assert question["key_facts"] == [], question["id"]
            assert question["expected_answer"] is None, question["id"]


def test_answerable_questions_have_ground_truth(dataset):
    for question in dataset:
        if not question["should_abstain"]:
            assert question["supporting_sources"], f"{question['id']} has no supporting_sources"
            assert question["key_facts"], f"{question['id']} has no key_facts"
            assert question["expected_answer"], f"{question['id']} has no expected_answer"


def test_supporting_source_structure_is_valid(dataset):
    for question in dataset:
        for source in question["supporting_sources"]:
            assert "document_id" in source
            assert "filename" in source
            assert "page_number" in source
            assert "chunk_id" in source


def test_follow_up_questions_have_valid_depends_on(dataset):
    seen_ids = set()

    for question in dataset:
        if question["category"] == "follow_up":
            assert question["depends_on"] is not None, question["id"]
            assert question["depends_on"] in seen_ids, (
                f"{question['id']} depends_on '{question['depends_on']}', which must appear "
                "earlier in the dataset so the evaluation runner can resolve it in order"
            )
        seen_ids.add(question["id"])


def test_multiple_documents_are_represented(dataset):
    document_ids = {
        source["document_id"]
        for question in dataset
        for source in question["supporting_sources"]
    }
    assert len(document_ids) >= 2, (
        "Dataset should reference more than one document to exercise multi-document questions"
    )


def test_multi_document_questions_reference_multiple_documents(dataset):
    for question in dataset:
        if question["category"] == "multi_document":
            document_ids = {source["document_id"] for source in question["supporting_sources"]}
            assert len(document_ids) >= 2, (
                f"{question['id']} is categorized multi_document but only references "
                f"{len(document_ids)} distinct document(s)"
            )
