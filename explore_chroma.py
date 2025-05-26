import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime
import uuid

# Initialize ChromaDB client and embedding model
@st.cache_resource
def init_chroma_and_embedder():
    client = chromadb.PersistentClient(path="./chroma_db")
    embedder = SentenceTransformer("all-distilroberta-v1")
    return client, embedder

client, embedder = init_chroma_and_embedder()

# Streamlit app
st.title("🔍 ChromaDB Semantic Search Explorer")

# Display collection statistics
st.markdown("## Collection Overview")
collections = client.list_collections()
if not collections:
    st.warning("No collections found in the ChromaDB database.")
else:
    stats = []
    for collection in collections:
        data = collection.get()
        doc_count = len(data["documents"])
        max_chunk_size = max((len(doc) for doc in data["documents"]), default=0)
        stats.append({
            "Collection Name": collection.name,
            "Document Count": doc_count,
            "Max Chunk Size (chars)": max_chunk_size
        })
    st.table(pd.DataFrame(stats))

# Semantic search in a selected collection
st.markdown("## Semantic Search in Selected Collection")

selected_collection_name = st.selectbox(
    "Select a collection to search",
    ["-- Select a collection --"] + [c.name for c in collections]
)

query = st.text_input("Enter your search query:", placeholder="e.g., machine learning advancements")
max_results = st.slider("Max results", 1, 10, 5)

if query and selected_collection_name and selected_collection_name != "-- Select a collection --":
    try:
        with st.spinner("Generating query embedding..."):
            query_embedding = embedder.encode(query).tolist()

        selected_collection = client.get_collection(name=selected_collection_name)
        results = selected_collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            include=["documents", "metadatas", "distances"]
        )

        results_list = []
        for doc, _id, distance in zip(
            results["documents"][0], results["ids"][0], results["distances"][0]
        ):
            results_list.append({
                "id": _id,
                "document": doc[:1000],
                "similarity": 1 - distance
            })

        results_list = sorted(results_list, key=lambda x: x["similarity"], reverse=True)

        if not results_list:
            st.info("No results found.")
        else:
            for i, result in enumerate(results_list, 1):
                with st.expander(f"Result {i} (Similarity: {result['similarity']:.4f})"):
                    st.markdown(f"**ID**: `{result['id']}`")
                    st.code(result["document"], language="markdown")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

elif query:
    st.warning("Please select a collection to search.")

# Collection preview
st.markdown("## Collection Preview")
selected = st.selectbox("Choose a collection to preview", [""] + [c.name for c in collections])
if selected:
    collection = client.get_collection(name=selected)
    limit = st.slider("Number of documents to preview", 1, 20, 5, key="preview_limit")
    data = collection.get(limit=limit)
    
    if not data["documents"]:
        st.info("This collection is empty.")
    else:
        for doc, _id in zip(data["documents"], data["ids"]):
            with st.expander(f"Document ID: {_id}"):
                st.code(doc[:1000], language="markdown")

# Footer
st.markdown("---")
st.markdown(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Powered by xAI")