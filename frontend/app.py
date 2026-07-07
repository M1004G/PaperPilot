"""PaperPilot — simple Streamlit frontend for the multi-agent research assistant."""
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")
st.title("📄 PaperPilot — Autonomous Research Assistant")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
    st.session_state.title = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Upload a paper")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded is not None and st.button("Ingest paper", type="primary"):
        with st.spinner("Parsing PDF and building index..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.doc_id = data["doc_id"]
                st.session_state.title = data["title"]
                st.session_state.messages = []
                st.success(f"Ingested: {data['title']} ({data['num_pages']} pages)")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    if st.session_state.doc_id:
        st.info(f"**Active paper:**\n{st.session_state.title}\n\n`doc_id: {st.session_state.doc_id}`")

if not st.session_state.doc_id:
    st.write("👈 Upload a research paper PDF to get started.")
    st.stop()

doc_id = st.session_state.doc_id

tab_summary, tab_report, tab_gaps, tab_chat = st.tabs(
    ["🧾 Summary", "📑 Report", "🔍 Research Gaps", "💬 Chat"]
)

with tab_summary:
    if st.button("Generate summary"):
        with st.spinner("Summary Agent working..."):
            try:
                data = requests.get(f"{BACKEND_URL}/summary/{doc_id}", timeout=120).json()
                st.subheader("TL;DR")
                st.write(data["tldr"])

                st.subheader("Key Findings")
                for f in data["key_findings"]:
                    st.markdown(f"- {f}")

                st.subheader("Section Summaries")
                for s in data["section_summaries"]:
                    with st.expander(s["heading"]):
                        st.write(s["summary"])
            except Exception as e:
                st.error(f"Failed to get summary: {e}")

with tab_report:
    if st.button("Generate full report"):
        with st.spinner("Report Agent assembling report..."):
            try:
                resp = requests.get(f"{BACKEND_URL}/report/{doc_id}", timeout=180)
                resp.raise_for_status()
                st.session_state.report_text = resp.text
            except Exception as e:
                st.error(f"Failed to get report: {e}")

    if st.session_state.get("report_text"):
        st.markdown(st.session_state.report_text)
        st.download_button(
            "⬇️ Download report (.md)",
            data=st.session_state.report_text,
            file_name=f"report_{doc_id}.md",
            mime="text/markdown",
        )

with tab_gaps:
    if st.button("Analyze research gaps"):
        with st.spinner("Gap Analysis Agent working..."):
            try:
                data = requests.get(f"{BACKEND_URL}/gaps/{doc_id}", timeout=120).json()
                st.subheader("Author-Acknowledged Limitations")
                if data["author_acknowledged"]:
                    for g in data["author_acknowledged"]:
                        st.markdown(f"- {g}")
                else:
                    st.caption("None explicitly stated by the authors.")

                st.subheader("Inferred Gaps (Critical Review)")
                if data["inferred"]:
                    for g in data["inferred"]:
                        badge = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(g["confidence"], "⚪")
                        st.markdown(f"{badge} **{g['gap']}** _(confidence: {g['confidence']})_")
                        st.caption(f"Suggested direction: {g['direction']}")
                else:
                    st.caption("None identified.")
            except Exception as e:
                st.error(f"Failed to get gap analysis: {e}")

with tab_chat:
    st.caption("Ask questions grounded in the uploaded paper (RAG-based).")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.caption(f"[chunk {src['chunk_id']}] (score {src['score']}) — {src['preview']}...")

    query = st.chat_input("Ask something about the paper...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("RAG Agent retrieving + answering..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/chat", json={"doc_id": doc_id, "query": query}, timeout=120
                    ).json()
                    st.write(resp["answer"])
                    with st.expander("Sources"):
                        for src in resp["sources"]:
                            st.caption(f"[chunk {src['chunk_id']}] (score {src['score']}) — {src['preview']}...")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": resp["answer"], "sources": resp["sources"]}
                    )
                except Exception as e:
                    st.error(f"Chat failed: {e}")
