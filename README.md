This is a two-part system for creating and exploring a semantic search database using ChromaDB and Sentence Transformers, visualized with Streamlit.

1.  **Script 1 (Embedding & Storage):** Loads Markdown files, generates embeddings, and stores them in a ChromaDB collection.
2.  **Script 2 (Streamlit Explorer):** Provides a web interface to view ChromaDB collections, perform semantic searches, and preview documents.

Here's a `README.md` to explain both scipts: 

# 🧠 ChromaDB Semantic Search Engine: Ingestion & Explorer

This project provides a two-part system to build and interact with a semantic search engine powered by ChromaDB, Sentence Transformers for embeddings, and Streamlit for the user interface.

1.  **📚 `ingest_markdown.py`**: Loads Markdown documents from specified site folders, generates text embeddings, and stores them in ChromaDB collections.
2.  **🔍 `app.py` (Streamlit App)**: A web application to explore the created ChromaDB collections, perform semantic searches, and preview documents.

## 🌊 Workflow Overview

1.  **📝 Prepare Data (Manual Step):**
    *   Organize your Markdown (.md) files into sub-folders within a base directory (e.g., `./sites_markdown/`). Each sub-folder will become a separate ChromaDB collection.

2.  **➡️ Ingestion (Script 1: `ingest_markdown.py`)**:
    *   Scans the base directory for site-specific sub-folders containing Markdown files.
    *   For each site:
        *   Reads all `.md` files.
        *   Uses `SentenceTransformer("all-distilroberta-v1")` to generate embeddings for each document's content.
        *   Creates (or gets) a ChromaDB collection named after the site folder (e.g., `my_site_docs`).
        *   Adds the documents, their embeddings, and unique IDs to the collection.
    *   The ChromaDB data is persisted to disk (in `./chroma_db/`).

3.  **🔎 Exploration (Script 2: `app.py`)**:
    *   Starts a Streamlit web application.
    *   Connects to the existing ChromaDB instance (`./chroma_db/`).
    *   **Displays Collection Overview:** Shows a table of all available collections, their document counts, and maximum chunk sizes.
    *   **Semantic Search:** Allows users to select a collection, enter a search query, and retrieve the most semantically similar documents.
    *   **Collection Preview:** Lets users select a collection and preview a limited number of its documents.

## 📜 Scripts

### 1. 📚 Markdown Ingestion & Embedding (`ingest_markdown.py`)

This script processes Markdown files, generates embeddings, and stores them in ChromaDB.

**🎯 Purpose:**

*   To populate a ChromaDB vector database with textual content from Markdown files, making it searchable semantically.
*   Organizes documents into collections based on their source site/folder.

**✨ Features:**

*   **📂 Folder-based Organization:** Reads Markdown files from sub-folders within a base directory. Each sub-folder name becomes a collection name.
*   **🧠 Embedding Generation:** Uses `SentenceTransformer("all-distilroberta-v1")` for creating dense vector embeddings.
*   **💾 Persistent Storage:** Uses `chromadb.PersistentClient` to save the database to disk (`./chroma_db/`).
*   **➕ Idempotent Collection Creation:** Uses `get_or_create_collection` to avoid errors if collections already exist (though re-running will add duplicate documents if not managed separately).
*   **🆔 Unique Document IDs:** Assigns a UUID to each document added.
*   **📊 Progress Indication:** Shows a progress bar during embedding generation.

**📋 Prerequisites:**

*   🐍 Python 3.x
*   📦 Python libraries: `chromadb`, `sentence-transformers`
    ```bash
    pip install chromadb sentence-transformers
    ```

**🔧 Configuration:**

*   `BASE_DIRECTORY` (in `if __name__ == "__main__":`): Set this to the path of your root folder containing site-specific sub-folders with Markdown files. (Default: `./sites_markdown`)

**🚀 How to Use:**

1.  **💾 Save the script:** e.g., as `ingest_markdown.py`.
2.  **📚 Prepare Data:**
    *   Create a base directory (e.g., `sites_markdown`).
    *   Inside it, create sub-folders for each "site" or category of documents (e.g., `sites_markdown/blog_posts`, `sites_markdown/product_docs`).
    *   Place your `.md` files within these sub-folders.
3.  **⚙️ Configure `BASE_DIRECTORY`** in the script if it's not `./sites_markdown`.
4.  **▶️ Execute the script:**
    ```bash
    python ingest_markdown.py
    ```
    *   This will create/populate the `./chroma_db` directory.

---

### 2. 🔍 Streamlit Semantic Search Explorer (`app.py`)

A web application to interact with the ChromaDB collections created by the ingestion script.

**🎯 Purpose:**

*   Provide a user-friendly interface to perform semantic searches on the embedded documents.
*   Allow exploration and preview of the content stored in ChromaDB.

**✨ Features:**

*   **📊 Collection Overview:** Lists all ChromaDB collections with basic statistics (document count, max chunk size).
*   **🔎 Semantic Search:**
    *   Select a collection.
    *   Enter a natural language query.
    *   Specify the maximum number of results.
    *   Displays results sorted by similarity, showing document snippets and similarity scores.
*   **📄 Collection Preview:** Allows browsing a sample of documents from a selected collection.
*   **⚡️ Caching:** Uses `@st.cache_resource` for efficient initialization of ChromaDB client and embedder.
*   **✨ User-Friendly UI:** Built with Streamlit for an interactive experience.

**📋 Prerequisites:**

*   🐍 Python 3.x
*   📦 Python libraries: `streamlit`, `chromadb`, `sentence-transformers`, `pandas`
    ```bash
    pip install streamlit chromadb sentence-transformers pandas
    ```
*   ✅ **ChromaDB Data:** Requires the `./chroma_db` directory to be populated by the `ingest_markdown.py` script first.

**🚀 How to Use:**

1.  **💾 Save the script:** e.g., as `app.py`.
2.  **✅ Ensure Data Exists:** Make sure you have run `ingest_markdown.py` and the `./chroma_db` directory exists and contains data.
3.  **▶️ Run the Streamlit App:**
    Open a terminal or command prompt, navigate to the script's directory, and run:
    ```bash
    streamlit run app.py
    ```
    *   This will open the web application in your default browser.

## ⚙️ Running the Full Pipeline

1.  **🛠️ Setup:**
    *   Install Python.
    *   Install all required Python dependencies:
        ```bash
        pip install streamlit chromadb sentence-transformers pandas
        ```
    *   Save both Python scripts (`ingest_markdown.py` and `app.py`) in the same directory.
2.  **📚 Prepare Your Markdown Data:**
    *   Create a base directory (e.g., `sites_markdown`).
    *   Organize your `.md` files into sub-folders within this base directory (e.g., `sites_markdown/site_A`, `sites_markdown/site_B`).
3.  **➡️ Run Ingestion Script:**
    *   Modify `BASE_DIRECTORY` in `ingest_markdown.py` if needed.
    *   Execute: `python ingest_markdown.py`
    *   This will create/update the `./chroma_db` folder.
4.  **🔍 Launch Streamlit Explorer:**
    *   Execute: `streamlit run app.py`
    *   Interact with your semantic search engine via the web browser.
