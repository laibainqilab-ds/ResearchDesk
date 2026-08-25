from app.rag import RAG


rag = RAG()

question = "What is Evo 2 used for?"

answer = rag.answer(question)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)