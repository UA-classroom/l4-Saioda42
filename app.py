import os
from dotenv import load_dotenv
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

st.set_page_config(page_title="Cypher System Rules Assistant", layout="wide")
st.title("⚔️ Cypher System Rules Assistant")
st.caption("Ask questions about the Cypher System rulebook")


@st.cache_resource
def get_db():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    db = chromadb.PersistentClient(path="chroma_db")
    collection = db.get_collection("cypher_rules")
    return model, collection


@st.cache_resource
def get_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        st.error("Missing GROQ_API_KEY in .env")
        st.stop()
    return Groq(api_key=key)


def search_rules(query, model, collection):
    embeddings = model.encode(query)

    results = collection.query(
        query_embeddings=[embeddings.tolist()],
        n_results=6,
        include=["documents", "distances", "metadatas"]
    )

    docs = results["documents"][0] if results["documents"] else []
    dists = results["distances"][0] if results["distances"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    return docs, dists, metas


@st.cache_data(ttl=3600)
def ask_llm(query, sources_tuple, _client):
    sources = list(sources_tuple)
    if not sources:
        return "❌ No relevant sources found for this query."

    context = "\n\n".join([f"[{i+1}] {src}" for i, src in enumerate(sources)])

    prompt = f"""You are a helpful assistant for the Cypher System RPG rulebook.
Answer the question clearly and in your own words, based on the sources below.
If the sources don't contain enough information, say so briefly.

SOURCES:
{context}

QUESTION: {query}

Answer:"""

    try:
        msg = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        return msg.choices[0].message.content
    except Exception as e:
        if "429" in str(e):
            return "⏳ Rate limited. Try again in a moment."
        return f"Error: {str(e)[:80]}"


def main():
    model, db = get_db()
    llm = get_llm()

    col1 = st.columns(1)[0]

    with col1:
        user_query = st.text_input("What do you want to know?")

    if user_query:
        with st.spinner("Looking it up..."):
            sources, scores, metadata = search_rules(user_query, model, db)
            answer = ask_llm(user_query, tuple(sources), llm)

        st.divider()
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for i, (src, score, meta) in enumerate(zip(sources, scores, metadata)):
                book = meta.get("source_file", "?").split("-")[0] if isinstance(meta, dict) else "?"
                with st.expander(f"{book} • {max(0, 1 - score):.0%} match"):
                    st.caption(src[:800] + "..." if len(src) > 800 else src)


if __name__ == "__main__":
    main()
