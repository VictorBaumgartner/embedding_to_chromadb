import os
import re
import uuid
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

def load_markdown_files(base_dir):
    site_docs = {}
    for site_folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, site_folder)
        if os.path.isdir(full_path):
            texts = []
            for root, _, files in os.walk(full_path):
                for file in files:
                    if file.endswith(".md"):
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                texts.append(content)
            if texts:
                site_docs[site_folder] = texts
    return site_docs

def embed_and_store(docs_by_site):
    client = PersistentClient(path="./chroma_db")
    model = SentenceTransformer("all-distilroberta-v1")

    for site_name_raw, docs in docs_by_site.items():
        print(f"Processing {site_name_raw} -> collection `{site_name_raw}`")

        # Embed using SentenceTransformer
        embeddings = model.encode(docs, show_progress_bar=True)

        collection = client.get_or_create_collection(name=site_name_raw)
        collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=[str(uuid.uuid4()) for _ in docs]
        )

if __name__ == "__main__":
    BASE_DIRECTORY = "./sites_markdown"
    documents = load_markdown_files(BASE_DIRECTORY)
    embed_and_store(documents)
