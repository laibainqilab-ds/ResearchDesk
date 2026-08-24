import streamlit as st


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


if page == "Chat":
    st.header("Chat")
    st.info("Chat functionality will be implemented in later phases.")


elif page == "Documents":
    st.header("Documents")
    st.info("Document ingestion will be implemented in Phase 2.")


elif page == "Retrieval Inspector":
    st.header("Retrieval Inspector")
    st.info("Retrieval inspection will be implemented in later phases.")


elif page == "Evaluation":
    st.header("Evaluation")
    st.info("Evaluation functionality will be implemented in Phase 5.")