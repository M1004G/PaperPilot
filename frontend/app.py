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

tab_summary, tab_report, tab_gaps, tab_repro, tab_chat = st.tabs(
    ["🧾 Summary", "📑 Report", "🔍 Research Gaps", "🧪 Reproducibility", "💬 Chat"]
)

with tab_summary:
    if st.button("Generate summary"):
        with st.spinner("Summary Agent working..."):
            try:
                data = requests.get(f"{BACKEND_URL}/summary/{doc_id}", timeout=120).json()
                st.subheader("Concise Overview")
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

with tab_repro:
    st.caption(
        "If the paper links a real GitHub repo, checks its hygiene (README, license, pinned deps, "
        "tests, CI) plus whether it plausibly matches the paper's claims. If not, PaperPilot "
        "generates an implementation attempt from the methodology and evaluates that instead — "
        "either way, one score for how much you should trust the code."
    )
    repo_url_override = st.text_input(
        "Repo URL (optional — overrides auto-detection / triggers a repo check instead of generation)",
        placeholder="https://github.com/owner/repo",
        key="repro_repo_url",
    )
    if st.button("Run reproducibility check"):
        with st.spinner("Working... this may take a minute if generating code."):
            try:
                params = {"repo_url": repo_url_override} if repo_url_override else {}
                st.session_state["repro_data"] = requests.get(
                    f"{BACKEND_URL}/reproducibility/{doc_id}", params=params, timeout=180
                ).json()
            except Exception as e:
                st.error(f"Failed to run reproducibility check: {e}")

    data = st.session_state.get("repro_data")
    if data:
        if data.get("note") and not data.get("checks"):
            st.warning(data["note"])
        else:
            if data.get("mode") == "repo_check":
                meta = data["repo_metadata"]
                st.subheader(f"[{meta['full_name']}]({data['repo_url']})")
            else:
                st.subheader("Generated implementation (no repo was linked)")
                st.caption("PaperPilot wrote this from the paper's methodology section — review before trusting it.")

            score = data["score"]
            st.metric("Reproducibility score", f"{score}/100", data["verdict"])
            st.progress(score / 100)

            st.markdown("**Static checks**")
            icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "na": "➖"}
            for c in data["checks"]:
                st.markdown(f"{icon.get(c['status'], '•')} **{c['label']}** — {c['detail']}")

            st.markdown("**Claim verification (paper vs. code)**")
            if data.get("claims"):
                verdict_icon = {"matches": "✅", "unclear": "⚠️", "not_evident": "❌"}
                for item in data["claims"]:
                    st.markdown(f"{verdict_icon.get(item['verdict'], '•')} **{item['claim']}** — {item['evidence']}")
            else:
                st.caption("No implementation claims were checked.")

            if data.get("mode") == "generated" and data.get("files"):
                st.markdown("---")
                st.markdown(data.get("gap_report") or "")
                st.markdown("### Generated Code")

                try:
                    zip_resp = requests.get(f"{BACKEND_URL}/reproducibility/{doc_id}/download", timeout=30)
                    st.download_button(
                        "⬇️ Download all files (.zip)", data=zip_resp.content,
                        file_name=f"paperpilot_{doc_id}_generated.zip", mime="application/zip",
                    )
                except Exception as e:
                    st.caption(f"(Download unavailable: {e})")

                lang_by_ext = {".py": "python", ".txt": "text", ".md": "markdown", ".json": "json", ".yml": "yaml", ".yaml": "yaml"}
                for filename in sorted(data["files"]):
                    with st.expander(f"📄 {filename}"):
                        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
                        st.code(data["files"][filename], language=lang_by_ext.get(ext, "text"))

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
