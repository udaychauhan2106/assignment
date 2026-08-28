import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import build_graph, run_turn


st.set_page_config(page_title="Aster & Row Support Assistant", page_icon="A")
st.title("Aster & Row Support Assistant")
st.caption("Customer support answers grounded in Aster & Row's supplied policies and order data.")

if "support_graph" not in st.session_state:
    st.session_state.support_graph = build_graph()
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit-{id(st.session_state)}"
if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.header("Conversation")
    if st.button("Clear conversation"):
        st.session_state.chat = []
        st.session_state.session_id = f"streamlit-{id(st.session_state)}-new"
        st.rerun()

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            st.caption("Sources")
            for source in message["sources"]:
                st.caption(f"{source['source']} - {source['heading']}")
        if message.get("handoff"):
            st.warning("Human support is recommended for this request.")

question = st.chat_input("Ask about policies, products, or an order")
if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the supplied information..."):
            try:
                response = run_turn(st.session_state.support_graph, st.session_state.session_id, question)
            except Exception:
                st.error("The assistant could not complete that request. Please try again or contact human support.")
                st.stop()
        st.write(response.answer)
        if response.sources:
            st.caption("Sources")
            for source in response.sources:
                st.caption(f"{source['source']} - {source['heading']}")
        if response.handoff:
            st.warning("Human support is recommended for this request.")
    st.session_state.chat.append({
        "role": "assistant",
        "content": response.answer,
        "sources": response.sources,
        "handoff": response.handoff,
    })
