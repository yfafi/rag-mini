import streamlit as st, tempfile, os
from rag_chain import ingest, build_chain

st.set_page_config(page_title="RAG-mini", page_icon="📄")
st.title("📄🔎 RAG mini-app")

uploaded = st.file_uploader("Charge PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
if st.button("Indexer") and uploaded:
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for f in uploaded:
            fp = os.path.join(tmp, f.name)
            with open(fp, "wb") as out: out.write(f.read())
            paths.append(fp)
        ingest(paths)
    st.success("Index construit !")

chain = build_chain()

if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

q = st.chat_input("Pose ta question…")
debug = st.sidebar.checkbox("Debug context")

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    st.chat_message("user").write(q)

    with st.spinner("🔍 Récupération…"):
        res = chain(q)
        answer = res["result"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

        if debug:
            st.sidebar.write("### Passages récupérés")
            for d in res["source_documents"]:
                st.sidebar.write(d.page_content[:400] + "…")
