import os

import streamlit as st
from dotenv import load_dotenv

from auth import authenticate, init_db, register_user
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
# Authentication UI
# ------------------------------------------------------------------ #

def render_auth_page() -> None:
    st.title("📋 Leave Policy Assistant")
    st.caption("Please log in or create an account to continue.")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        with st.form("register_form", clear_on_submit=False):
            full_name = st.text_input("Full name")
            email = st.text_input("Email")
            new_username = st.text_input("Username", key="register_username")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Register", use_container_width=True)

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = register_user(new_username, email, full_name, new_password)
                (st.success if ok else st.error)(message)


def logout() -> None:
    for key in ("user", "messages"):
        st.session_state.pop(key, None)
    st.rerun()


# ------------------------------------------------------------------ #
# Streamlit UI
# ------------------------------------------------------------------ #

def main() -> None:
    st.set_page_config(
        page_title="Leave Policy Chatbot",
        page_icon="📋",
        layout="centered",
    )

    init_db()

    if "user" not in st.session_state:
        render_auth_page()
        return

    user = st.session_state["user"]

    st.title("📋 Leave Policy Assistant")
    st.caption("Ask anything about Flairminds Software's leave policy.")

    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY is missing. Add it to a `.env` file and restart.")
        st.stop()

    engine = get_engine()

    # ---- Sidebar --------------------------------------------------- #
    with st.sidebar:
        st.markdown(f"👤 **{user['full_name']}**")
        st.caption(f"@{user['username']} · {user['email']}")
        if st.button("🚪 Logout"):
            logout()

        st.divider()
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
