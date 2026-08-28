import streamlit as st

from app.rag import RAG


st.set_page_config(
    page_title="ResearchDesk",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 ResearchDesk")
st.write("Document Research Assistant")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Chat",
        "Documents",
        "Retrieval Inspector",
        "Evaluation",
    ],
)


if "rag" not in st.session_state:
    st.session_state.rag = RAG()

if "messages" not in st.session_state:
    st.session_state.messages = []


if page == "Chat":
    st.header("Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                st.caption("Sources:")

                for source in message["sources"]:
                    if source["page_number"] is not None:
                        st.caption(
                            f"- {source['filename']}, "
                            f"page {source['page_number']}, "
                            f"chunk {source['chunk_id']}"
                        )
                    else:
                        st.caption(
                            f"- {source['filename']}, "
                            f"chunk {source['chunk_id']}"
                        )

    question = st.chat_input("Ask a question about your documents")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        conversation_history = st.session_state.messages[:-1]

        result = st.session_state.rag.answer(
            question=question,
            conversation_history=conversation_history,
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )

        st.rerun()


elif page == "Documents":
    st.header("Documents")
    st.info("Document management will be expanded in later phases.")


elif page == "Retrieval Inspector":
    st.header("Retrieval Inspector")
    st.info("Retrieval inspection will be implemented in Phase 4.")


elif page == "Evaluation":
    st.header("Evaluation")
    st.info("Evaluation functionality will be implemented in Phase 5.")