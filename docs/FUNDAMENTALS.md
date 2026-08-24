# ResearchDesk Fundamentals

## 1. Embeddings

An embedding is a numerical representation of text. It converts text into a vector of numbers so that a computer can compare the meaning of different pieces of text.

In ResearchDesk, embeddings are used to represent both documents and user questions. We can then compare their vectors to find text that is semantically similar to the question.

For example, in the semantic search experiment, the query:

"How do computers learn from data?"

was most similar to:

"Machine learning allows computers to learn patterns from data."

This showed that semantic search can find related meaning even when the exact words are not identical.

### Experiment

I used the Sentence Transformers model `BAAI/bge-small-en-v1.5` to generate embeddings for several text examples. I then compared the query embedding with the text embeddings and ranked the results by similarity.

## 2. Semantic Search

Semantic search finds information based on the meaning of the text rather than only looking for exact matching words.

In the semantic search experiment, the question "How do computers learn from data?" retrieved "Machine learning allows computers to learn patterns from data." as the most similar result even though the wording was different. This showed how embeddings allow the system to find related meaning.

## 3. Cosine Similarity

Cosine similarity is used to measure how similar two text embeddings are. The embeddings are represented as vectors, and the similarity score helps determine how closely related the meanings of two pieces of text are.

In my experiment, the results were ranked by their similarity scores. The highest score represented the text that was most semantically similar to the query.

## 4. Top-K Retrieval

Top-K retrieval means selecting the K highest-scoring results from a retrieval operation. For example, if Top-K is set to 3, the system keeps the three most relevant chunks instead of passing every available chunk to the language model.

Top-K is important because a document collection can contain a large number of chunks, while only a smaller number may be relevant to a particular question. Selecting the most relevant chunks also helps reduce unnecessary information sent to the LLM.

## 5. Chunking

Chunking is the process of breaking a large document into smaller pieces of text called chunks. RAG systems use chunks because retrieving smaller relevant sections is more useful than sending an entire large document to the LLM.

In the chunking experiment, I divided a text into chunks using a fixed chunk size. I observed that the way a document is divided can affect whether an idea or sentence stays together or gets split between chunks.

Chunk size should therefore not be selected randomly. It should be tested to determine whether the resulting chunks contain useful and meaningful information.

## 6. Chunk Overlap

Chunk overlap means repeating some text from the end of one chunk at the beginning of the next chunk.

I tested a chunk size of 20 words with an overlap of 5 words and compared it with an overlap of 0. With no overlap, information could be split directly between two chunks. With overlap, some surrounding context was preserved between neighboring chunks.

Overlap can therefore help prevent important context from being lost at chunk boundaries. However, too much overlap can create unnecessary duplicate information.

## 7. Metadata

Metadata is information that describes a piece of data. In ResearchDesk, metadata is attached to each document chunk so that the system can identify where the chunk came from.

In the metadata experiment, I attached information such as the document ID, filename, file type, page number, and chunk ID to a piece of text.

Metadata is important because it allows retrieved evidence to be traced back to its original document and page. This will later allow ResearchDesk to provide citations based on real source information instead of allowing the LLM to invent filenames or page numbers.

## 8. Context Windows

A context window is the amount of information an LLM can process as input during a request. This can include the user's question, conversation history, instructions, and retrieved document chunks.

In the context experiment, I started with five chunks containing 35 words and then selected only two chunks, containing 17 words, to represent the information sent to the LLM.

This showed why retrieval is important. Instead of sending every available document chunk to the LLM, the system should select the most relevant evidence. This reduces unnecessary context and helps the model focus on information related to the question.

## 9. Hallucinations

A hallucination occurs when an LLM generates information that is unsupported or not present in the available evidence.

This is a major concern for ResearchDesk because the system is supposed to answer questions using the user's documents. If the documents do not contain enough information to answer a question, the system should not allow the LLM to simply guess.

Instead, ResearchDesk should explicitly abstain and tell the user that there is not enough information in the provided documents.

This is why grounding the answer in retrieved evidence is an important part of the RAG system.

## 10. Structured Output

Structured output means requiring an LLM to return information in a predefined format instead of unrestricted text.

For example, instead of returning a sentence such as "This looks like a research question," a model could return a structured result such as a route field containing "research".

Structured output makes LLM responses easier for the application to parse and validate. ResearchDesk can use Pydantic models to define and validate important structured outputs.

## 11. Tool Calling

Tool calling allows an LLM to request that the application execute a predefined function.

For example, ResearchDesk could provide a search_documents tool. When the LLM determines that it needs information from the user's documents, it can request the tool, and the application performs the actual search.

The LLM does not directly access the database. The application controls the available tools and executes them. This keeps operations controlled and separates the LLM's decision-making from the actual implementation.

## 12. Chain vs Agent

A chain follows a predefined sequence of steps. For example, a RAG chain could always follow the sequence: rewrite the question, retrieve documents, and generate an answer.

An agent can make decisions about which action to take within a controlled workflow. For example, an agent might decide to search for more evidence if the first retrieval attempt is insufficient.

An agent should therefore only be introduced when decision-making is actually useful. If a deterministic sequence can solve the problem, a normal function or chain is usually simpler and easier to debug.