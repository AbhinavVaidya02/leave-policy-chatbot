import os

import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine, INDEX_DIR

load_dotenv()

PDF_PATH = "Leave Policy.pdf"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


# ------------------------------------------------------------------ #
# Engine initialisation (cached across reruns via session_state)
# ------------------------------------------------------------------ #

def get_engine() -> RAGEngine:
    if "rag_engine" not in st.session_state:
        engine = RAGEngine(api_key=GOOGLE_API_KEY)
        if not engine.load_index(INDEX_DIR):
            st.error(
                "Index not found. Run the ingestion script first:\n\n"
                "```\npython ingest.py\n```"
            )
            st.stop()
        st.session_state["rag_engine"] = engine
    return st.session_state["rag_engine"]


# ------------------------------------------------------------------ #
# Streamlit UI
# ------------------------------------------------------------------ #

def main() -> None:
    st.set_page_config(
        page_title="Leave Policy Chatbot",
        page_icon="📋",
        layout="centered",
    )

    st.title("📋 Leave Policy Assistant")
    st.caption("Ask anything about Flairminds Software's leave policy.")

    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY is missing. Add it to a `.env` file and restart.")
        st.stop()

    engine = get_engine()

    # ---- Sidebar --------------------------------------------------- #
    with st.sidebar:
        st.header("Controls")
        if st.button("🗑️ Clear chat"):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        st.markdown("**Source document**")
        st.markdown(f"`{PDF_PATH}`")
        st.caption("To rebuild the index, run `python ingest.py`.")

    # ---- Chat history ---------------------------------------------- #
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ---- Input ----------------------------------------------------- #
    if user_input := st.chat_input("Ask about leave policy…"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Pass history excluding the message just appended
                history = st.session_state["messages"][:-1]
                answer = engine.chat(query=user_input, chat_history=history)
            st.write(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
