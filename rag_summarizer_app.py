import os
import re
import time
import logging
import tempfile
from typing import List

import streamlit as st
import PyPDF2
import numpy as np
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer


# ============================================================
# Suppress warnings
# ============================================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# ============================================================
# RAG Summarizer Class
# ============================================================
class RAGSummarizer:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        llm_model: str = "google/flan-t5-large",
    ):

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Embedding model
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

        # FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)

        self.chunks = []
        self.metadata = []

        # LLM
        self.llm = pipeline(
            "text2text-generation",
            model=llm_model,
            max_length=512,
            device=-1
        )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

    # ============================================================
    # 1️⃣ Ingest Documents
    # ============================================================
    def ingest_documents(self, file_paths: List[str], file_name_map=None):

        self.chunks = []
        self.metadata = []

        for file_path in file_paths:

            # Use real filename instead of temp name
            real_name = (
                file_name_map[file_path]
                if file_name_map and file_path in file_name_map
                else os.path.basename(file_path)
            )

            text = ""

            if file_path.endswith(".pdf"):
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            page_text = re.sub(r"\s+", " ", page_text)
                            text += page_text + "\n"

            elif file_path.endswith((".txt", ".md")):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = re.sub(r"\s+", " ", file.read())

            if not text.strip():
                continue

            split_chunks = self.text_splitter.split_text(text)

            for i, chunk in enumerate(split_chunks):
                self.chunks.append(chunk)
                self.metadata.append(f"{real_name} | chunk {i+1}")

        if not self.chunks:
            raise ValueError("No valid text found in uploaded documents.")

    # ============================================================
    # 2️⃣ Create Embeddings
    # ============================================================
    def create_embeddings(self):
        embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        self.index.reset()
        self.index.add(embeddings)

    # ============================================================
    # 3️⃣ Retrieve Chunks
    # ============================================================
    def retrieve_chunks(self, query: str, top_k: int = 5):

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        top_k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                results.append({
                    "chunk": self.chunks[idx],
                    "source": self.metadata[idx],
                    "score": float(distances[0][i])
                })

        return results

    # ============================================================
    # 4️⃣ Answer Question
    # ============================================================
    def answer_question(self, question: str, top_k: int = 5):

        start_time = time.time()
        retrieved = self.retrieve_chunks(question, top_k)

        if not retrieved:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n".join(
            [f"[Source: {r['source']}]\n{r['chunk']}" for r in retrieved]
        )

        prompt = f"""
Use the context below to answer the question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm(prompt)[0]["generated_text"]
        latency = time.time() - start_time

        return {
            "answer": response,
            "sources": list(set([r["source"] for r in retrieved])),
            "retrieved_chunks": retrieved,
            "latency": latency
        }

    # ============================================================
    # 5️⃣ Full Pipeline
    # ============================================================
    def process_documents(self, file_paths, question, file_name_map=None):
        self.ingest_documents(file_paths, file_name_map)
        self.create_embeddings()
        return self.answer_question(question)


# ============================================================
# Streamlit App
# ============================================================

summarizer = RAGSummarizer()

st.set_page_config(
    page_title="RAG Multi-Document QA",
    layout="wide"
)

st.title("📄 Multi-Paper Research QA (RAG)")
st.write("Upload multiple research papers and ask cross-paper questions.")

uploaded_files = st.file_uploader(
    "Upload Research Papers",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

query = st.text_input(
    "Ask a question",
    value="What are the main contributions of these papers?"
)

if uploaded_files and query:

    if st.button("✨ Get Answer"):

        with st.spinner("Processing documents..."):

            file_paths = []
            file_name_map = {}

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp:

                    tmp.write(uploaded_file.read())
                    file_paths.append(tmp.name)

                    # Map temp path → real filename
                    file_name_map[tmp.name] = uploaded_file.name

            try:
                result = summarizer.process_documents(
                    file_paths,
                    query,
                    file_name_map
                )

                st.subheader("📝 Answer")
                st.write(result["answer"])

                st.subheader("📚 Papers Used")
                for src in result["sources"]:
                    st.write("•", src)

                st.subheader("📑 Retrieved Chunks")
                for i, r in enumerate(result["retrieved_chunks"], 1):
                    with st.expander(f"Chunk {i} ({r['source']})"):
                        st.write(r["chunk"])
                        st.write("Similarity Score:", round(r["score"], 4))

                st.subheader("📊 Metrics")
                st.write("Processing Time:", round(result["latency"], 2), "seconds")

            finally:
                for path in file_paths:
                    os.remove(path)

else:
    st.info("Upload research papers and enter a question.")