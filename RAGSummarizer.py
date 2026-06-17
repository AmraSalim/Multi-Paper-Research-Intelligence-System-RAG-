import os
import re
import time
import logging
from typing import List, Tuple

import PyPDF2
import numpy as np
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class RAGSummarizer:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        llm_model: str = "google/flan-t5-large",
    ):
        """
        Multi-document RAG Question Answering System
        """

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Embedding model
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

        # FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)

        # Storage
        self.chunks = []
        self.metadata = []

        # LLM for QA
        self.llm = pipeline(
            "text2text-generation",
            model=llm_model,
            max_length=512,
            device=-1  # CPU
        )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

    # ============================================================
    # 1️⃣ Ingest Multiple Documents
    # ============================================================
    def ingest_documents(self, file_paths: List[str]) -> None:
        """
        Read multiple files and create chunks with metadata.
        """
        self.chunks = []
        self.metadata = []

        for file_path in file_paths:
            print(f"Reading file: {file_path}")

            text = ""

            if file_path.endswith(".pdf"):
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            page_text = re.sub(r"\s+", " ", page_text)
                            text += page_text + "\n"

            elif file_path.endswith((".txt", ".md")):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    text = re.sub(r"\s+", " ", text)

            else:
                print(f"Skipping unsupported file: {file_path}")
                continue

            if not text.strip():
                continue

            # Split into chunks
            split_chunks = self.text_splitter.split_text(text)

            for chunk in split_chunks:
                self.chunks.append(chunk)
                self.metadata.append(os.path.basename(file_path))

        if not self.chunks:
            raise ValueError("No valid text found in uploaded documents.")

        print(f"Total chunks created: {len(self.chunks)}")

    # ============================================================
    # 2️⃣ Create Embeddings
    # ============================================================
    def create_embeddings(self) -> None:
        embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        self.index.reset()
        self.index.add(embeddings)
        print(f"Embeddings shape: {embeddings.shape}")

    # ============================================================
    # 3️⃣ Retrieve Relevant Chunks
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
    # 4️⃣ Answer Question Using Retrieved Context
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

        try:
            response = self.llm(prompt)[0]["generated_text"]
        except Exception as e:
            response = f"Error during generation: {str(e)}"

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
    def process_documents(self, file_paths: List[str], question: str):
        self.ingest_documents(file_paths)
        self.create_embeddings()
        return self.answer_question(question)