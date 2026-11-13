"""PaperPilot — simple Streamlit frontend for the multi-agent research assistant."""
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def api(method: str, path: str, **kwargs) -> requests.Response:
    """Call the backend. On an error status, raise with the backend's own message."""
    resp = requests.request(method, f"{BACKEND_URL}{path}", timeout=180, **kwargs)
    if not resp.ok:
        try:
            detail = resp.json()["detail"]
        except Exception:
            detail = resp.text
        raise RuntimeError(detail)
    return resp


def show_sources(sources: list[dict]):
    with st.expander("Sources"):
        for src in sources:
            st.caption(f"[{src['heading']}, p.{src['page_start']}] (score {src['score']}) — {src['preview']}...")


st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")
st.title("📄 PaperPilot — Autonomous Research Assistant")

# Everything below belongs to the active paper and is reset when a new one is ingested.
for key, default in {
    "doc_id": None, "title": None, "messages": [],
    "summary": None, "gaps": None, "report_text": None,
}.items():
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("1. Upload a paper")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded is not None and st.button("Ingest paper", type="primary"):
        with st.spinner("Parsing PDF and building index..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            try:
                data = api("POST", "/upload", files=files).json()
                st.session_state.update(
                    doc_id=data["doc_id"], title=data["title"],
                    messages=[], summary=None, gaps=None, report_text=None,
                )
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
                st.session_state.summary = api("GET", f"/summary/{doc_id}").json()
            except Exception as e:
                st.error(f"Failed to get summary: {e}")

    data = st.session_state.summary
    if data:
        st.subheader("Concise Overview")
        st.write(data["tldr"])

        st.subheader("Key Findings")
        for f in data["key_findings"]:
            st.markdown(f"- {f}")

        st.subheader("Section Summaries")
        for s in data["section_summaries"]:
            with st.expander(s["heading"]):
                st.write(s["summary"])

with tab_report:
    if st.button("Generate full report"):
        with st.spinner("Report Agent assembling report..."):
            try:
                st.session_state.report_text = api("GET", f"/report/{doc_id}").text
            except Exception as e:
                st.error(f"Failed to get report: {e}")

    if st.session_state.report_text:
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
                st.session_state.gaps = api("GET", f"/gaps/{doc_id}").json()
            except Exception as e:
                st.error(f"Failed to get gap analysis: {e}")

    data = st.session_state.gaps
    if data:
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

with tab_chat:
    st.caption("Ask questions grounded in the uploaded paper (RAG-based).")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                show_sources(msg["sources"])

    query = st.chat_input("Ask something about the paper...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("RAG Agent retrieving + answering..."):
                try:
                    resp = api("POST", "/chat", json={"doc_id": doc_id, "query": query}).json()
                    st.write(resp["answer"])
                    show_sources(resp["sources"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": resp["answer"], "sources": resp["sources"]}
                    )
                except Exception as e:
                    st.error(f"Chat failed: {e}")
