from app.rag import RAG


rag = RAG()

conversation = []

question1 = "What is Evo 2?"

result1 = rag.answer(
    question=question1,
    conversation_history=conversation,
)

print("\nQuestion:")
print(question1)

print("\nAnswer:")
print(result1["answer"])

conversation.append(
    {
        "role": "user",
        "content": question1,
    }
)

conversation.append(
    {
        "role": "assistant",
        "content": result1["answer"],
    }
)

question2 = "What is it used for?"

result2 = rag.answer(
    question=question2,
    conversation_history=conversation,
)

print("\nFollow-up Question:")
print(question2)

print("\nAnswer:")
print(result2["answer"])

print("\nSources:")

for source in result2["sources"]:
    if source["page_number"] is not None:
        print(
            f"- {source['filename']}, "
            f"page {source['page_number']}, "
            f"chunk {source['chunk_id']}"
        )
    else:
        print(
            f"- {source['filename']}, "
            f"chunk {source['chunk_id']}"
        )