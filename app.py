# Importing the Streamlit UI, PDF parsing, embedding, retrieval, and generation dependencies.
# These packages must be installed according to the project README.


import os
import glob
import time
import numpy as np
import pandas as pd
import re
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from pypdf.errors import PdfReadError, PdfStreamError
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


# PAGE CONFIGURATION & CONSTANTS
# ==============================================================================

st.set_page_config(
    page_title="Modular & Agentic RAG PDF Assistant",
    page_icon="📚",
    layout="wide",
)

PDF_FOLDER_PATH = "./PDF Files"
FAISS_MODULAR_PATH = "./faiss_modular_index"
os.makedirs(PDF_FOLDER_PATH, exist_ok=True)


# CACHED CORE MODELS & VECTOR STORE INITIALIZATION
# ==============================================================================

@st.cache_resource(show_spinner="Loading Embedding, Reranker, and Llama 3 Models...")
def load_core_models(): 
    """Loads and caches heavy neural models once per session."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    rerank_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    llm = Ollama(model="llama3")
    return embeddings, rerank_model, llm


def load_pdf_documents(pdf_path: str):
    """Safely extracts pages from a single PDF file."""
    # Convert PDF-reading failures into a visible warning instead of stopping the app.
    try:
        return PyPDFLoader(pdf_path).load()
    except (PdfReadError, PdfStreamError, OSError) as error:
        st.warning(f"Could not read PDF '{os.path.basename(pdf_path)}': {error}")
        return []


@st.cache_resource(show_spinner="Initializing FAISS & BM25 Hybrid Index...")
def initialize_knowledge_base(_embeddings, top_n: int = 6):
    """Builds or loads the Modular FAISS index and constructs the 3-Stage Hybrid Funnel."""
    # Split PDFs into overlapping chunks so text can be retrieved in useful context units.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Compare the corpus on disk with documents already stored in the FAISS index.
    current_pdfs = sorted(glob.glob(os.path.join(PDF_FOLDER_PATH, "*.pdf")))

    if os.path.exists(FAISS_MODULAR_PATH):
        vector_store = FAISS.load_local(
            FAISS_MODULAR_PATH,
            _embeddings,
            allow_dangerous_deserialization=True,
        )
        existing_pdfs = {
            os.path.normpath(doc.metadata.get("source", ""))
            for doc in vector_store.docstore._dict.values()
        }
        new_pdfs = [
            pdf for pdf in current_pdfs if os.path.normpath(pdf) not in existing_pdfs
        ]

        if new_pdfs:
            new_docs = []
            for pdf_path in new_pdfs:
                new_docs.extend(load_pdf_documents(pdf_path))
            if new_docs:
                new_chunks = text_splitter.split_documents(new_docs)
                vector_store.add_documents(new_chunks)
                vector_store.save_local(FAISS_MODULAR_PATH)
    else:
        if not current_pdfs:
            return None, None, 0
        all_docs = []
        for pdf_path in current_pdfs:
            all_docs.extend(load_pdf_documents(pdf_path))
        if not all_docs:
            return None, None, 0
        chunks = text_splitter.split_documents(all_docs)
        vector_store = FAISS.from_documents(chunks, _embeddings)
        vector_store.save_local(FAISS_MODULAR_PATH)

    all_chunks = list(vector_store.docstore._dict.values())
    return vector_store, all_chunks, len(current_pdfs)


def build_retrievers(vector_store, all_chunks, rerank_model, top_n: int = 6):
    """Constructs both the Modular 3-Stage Funnel and Naive Retriever for live comparison."""
    # Stage 1: retrieve diverse semantic candidates and keyword candidates.
    faiss_mmr = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}
    )
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 10

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_mmr], weights=[0.4, 0.6]
    )
    # Stage 2: combine both retrieval methods, then rerank the combined candidates.
    compressor = CrossEncoderReranker(model=rerank_model, top_n=top_n)
    modular_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hybrid_retriever
    )


    return modular_retriever


# AGENTIC CONTROLLER HELPERS (GATEKEEPER, REWRITER & FORMATTER)
# ==============================================================================
PROMPT_TEMPLATE = """Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)


def format_docs_with_sources(docs):
    """Formats retrieved Document objects with source filename and page number headers."""
    # Attach source metadata to each chunk so answers remain traceable to the PDFs.
    formatted_chunks = []
    for doc in docs:
        source_file = os.path.basename(doc.metadata.get("source", "Unknown PDF"))
        page_num = doc.metadata.get("page", 0) + 1
        formatted_chunks.append(
            f"[Source: {source_file}, Page: {page_num}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def rewrite_query_agent(llm, original_query: str) -> str:
    """Uses few-shot prompting to preserve entity names and core technical terms."""
    # Ask the local model to produce a concise search query while retaining key entities.
    rewrite_prompt = f"""You are a search query optimizer for a technical PDF archive.
Given a user question, generate a targeted 4 to 7 word search query.

RULES:
1. ALWAYS KEEP distinguishing paper names, versions, or authors (e.g., 'original RAG', 'Lewis', 'Llama 3', 'Transformer', 'NIST').
2. Include the core technical subject terms.
3. Remove conversational filler and question phrases (e.g., 'what are the', 'can you tell me').
4. Output ONLY the search query on a single line. Do not write explanations, quotes, or notes.

Example 1:
Question: What are the parameter sizes of the foundation models in the Llama 3 report?
Search Query: Llama 3 foundation models parameter sizes tokens

Example 2:
Question: According to the NIST framework, what are the characteristics of trustworthy AI?
Search Query: NIST AI Risk Management Framework trustworthy AI characteristics

Example 3:
Question: In the original Retrieval-Augmented Generation (RAG) paper, what are the two RAG model formulations proposed?
Search Query: original RAG Lewis models formulations RAG-Sequence RAG-Token

Question: {original_query}
Search Query:"""
    raw_rewrite = llm.invoke(rewrite_prompt).strip()
    lines = [line.strip() for line in raw_rewrite.split("\n") if line.strip()]
    first_line = lines[0] if lines else original_query
    clean = re.sub(r"\(.*?\)", "", first_line)
    clean = re.sub(
        r"^(search query:|keywords:|here are.*?:|\d+\.)",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = clean.replace('"', "").replace("'", "").strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean if clean else original_query


def grade_relevance_gate(llm, question: str, context: str) -> bool:
    """Evaluates whether retrieved chunks contain sufficient evidence to answer the question."""
    # The gate requires the model to return a strict YES/NO relevance verdict.
    gate_prompt = f"""You are a strict retrieval evaluator.
Does the following CONTEXT contain facts or details that directly help answer the QUESTION?
- If the context contains relevant facts or answers the question, reply YES.
- If the context is off-topic, only cites titles, or does NOT contain facts to answer the question, reply NO.
Output ONLY the single word YES or NO.

CONTEXT:
{context}

QUESTION: {question}
VERDICT (YES/NO):"""
    verdict = llm.invoke(gate_prompt).strip().upper()
    return "YES" in verdict and "NO" not in verdict[:4]


# SIDEBAR CONTROLS & PDF UPLOADER
# ==============================================================================
embeddings, rerank_model, llm = load_core_models()

with st.sidebar:
    # Display corpus controls and keep all PDF/index operations in the sidebar.
    st.title("⚙️ RAG Architecture Control")
    st.markdown("Configure the retrieval funnel and agentic verification settings.")

   

    top_n_chunks = st.slider(
        "Reranker Top-N Chunks:",
        min_value=3,
        max_value=8,
        value=6,
        help="Number of top chunks retained after Cross-Encoder neural reranking.",
    )

    enable_agent_gate = st.toggle(
        "Enable Agentic Gate & Query Rewriter",
        value=True,
        help="Verifies retrieved evidence before generation and rewrites queries on weak matches.",
    )

    st.divider()
    st.subheader("📂 PDF Corpus Manager")

    uploaded_files = st.file_uploader(
        "Upload New PDFs to Knowledge Base",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        # Save only newly uploaded PDFs, then invalidate caches and rebuild the index.
        new_file_saved = False
        for uf in uploaded_files:
            dest_path = os.path.join(PDF_FOLDER_PATH, uf.name)
            if not os.path.exists(dest_path):
                with open(dest_path, "wb") as f:
                    f.write(uf.getbuffer())
                new_file_saved = True
        if new_file_saved:
            st.cache_resource.clear()
            st.success("New PDF(s) saved! Re-indexing knowledge base...")
            st.rerun()

    vector_store, all_chunks, pdf_count = initialize_knowledge_base(
        embeddings, top_n=top_n_chunks
    )

    if vector_store is not None:
        st.metric("Indexed PDFs", pdf_count)
        st.metric("Total Structural Chunks", len(all_chunks))
        with st.expander("View Loaded PDF Files"):
            for pdf_file in sorted(glob.glob(os.path.join(PDF_FOLDER_PATH, "*.pdf"))):
                st.caption(f"📄 {os.path.basename(pdf_file)}")
    else:
        st.warning(f"No PDFs found in `{PDF_FOLDER_PATH}`. Upload a PDF above.")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# MAIN CHAT INTERFACE & EXECUTION PIPELINE
# ==============================================================================
st.title("📚 Modular & Agentic RAG Document Assistant")
st.caption(
    "Powered by Hybrid Search (FAISS MMR + BM25), Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`), and Local Llama 3 Agentic Gating"
)

# Initialize session chat history so previous questions and answers persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render each prior message, including timing and retrieved-source metadata when available.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            st.caption(
                f"⏱️ **Total:** {meta['total_time']:.2f}s "
                f"(Retrieval/Agent: {meta['ret_time']:.2f}s | LLM Generation: {meta['gen_time']:.2f}s) "
                f"| **Gate Passed:** `{meta['gate_passed']}`"
                + (
                    f" | **Rewritten Query:** `'{meta['rewritten_query']}'`"
                    if meta.get("rewritten_query")
                    else ""
                )
            )
            with st.expander(f"🔍 Inspect Retrieved Source Chunks ({len(meta['sources'])} chunks)"):
                for idx, src in enumerate(meta["sources"], 1):
                    st.markdown(
                        f"**Chunk {idx} — `{src['file']}` (Page {src['page']})**"
                    )
                    st.text(src["content"])
                    st.divider()

# Process a newly submitted chat prompt.
if user_query := st.chat_input("Ask a technical question about your uploaded PDFs..."):
    if vector_store is None:
        st.error("Please add PDF files to `./PDF Test Files` or upload them in the sidebar first.")
        st.stop()

    # Store and immediately display the user's question.
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    modular_retriever = build_retrievers(
        vector_store, all_chunks, rerank_model, top_n=top_n_chunks
    )

    with st.chat_message("assistant"):
        # Track whether a rewritten query was used and whether the evidence passed the gate.
        rewritten_query_used = None
        gate_passed = True

        # Modular Agentic RAG       
        with st.status("Running Modular & Agentic RAG Pipeline...", expanded=True) as status:
                t_ret_start = time.perf_counter()

                # Pass 1: retrieve and rerank context using the hybrid modular retriever.
                st.write("1️⃣ **Pass 1:** Fetching candidates via FAISS (MMR) + BM25 & Cross-Encoder Reranking...")
                docs = modular_retriever.invoke(user_query)
                context_str = format_docs_with_sources(docs)

                if enable_agent_gate:
                    # Verify whether the initially retrieved context is sufficient.
                    st.write("2️⃣ **Agentic Gatekeeper:** Verifying factual sufficiency of retrieved chunks...")
                    gate_passed = grade_relevance_gate(llm, user_query, context_str)

                    if not gate_passed:
                        # Pass 2: rewrite the weak query and retrieve evidence with the rewritten terms.
                        st.write("⚠️ **Gate Verdict: NO** — Initial context insufficient. Triggering Query Rewriter...")
                        rewritten_query_used = rewrite_query_agent(llm, user_query)
                        st.write(f"🔄 **Pass 2 Retry Query:** `{rewritten_query_used}`")

                        docs_retry = modular_retriever.invoke(rewritten_query_used)
                        context_retry = format_docs_with_sources(docs_retry)

                        if grade_relevance_gate(llm, user_query, context_retry):
                            st.write("✅ **Pass 2 Gate Verdict: YES** — Relevant evidence recovered!")
                            docs = docs_retry
                            context_str = context_retry
                            gate_passed = True
                        else:
                            st.write("🛑 **Pass 2 Gate Verdict: NO** — Topic out-of-scope. Abstaining safely.")
                            docs = docs_retry
                            context_str = context_retry
                            gate_passed = False
                    else:
                        st.write("✅ **Gate Verdict: YES** — High-confidence context verified on Pass 1.")

                ret_time = time.perf_counter() - t_ret_start

                if enable_agent_gate and not gate_passed:
                    # Abstain instead of generating an unsupported answer after both retrieval passes fail.
                    gen_time = 0.0
                    answer = (
                        "I don't know based on the provided documents. "
                        "The uploaded files do not contain information relevant to this question."
                    )
                else:
                    # Ground the local language model response in the formatted retrieved context.
                    st.write("3️⃣ **Llama 3 Generator:** Synthesizing grounded answer...")
                    t_gen_start = time.perf_counter()
                    formatted_prompt = prompt.invoke(
                        {"context": context_str, "question": user_query}
                    )
                    answer = StrOutputParser().invoke(llm.invoke(formatted_prompt))
                    gen_time = time.perf_counter() - t_gen_start

                total_time = ret_time + gen_time
                status.update(
                    label=f"Completed in {total_time:.2f}s (Retrieval/Agent: {ret_time:.2f}s | Generation: {gen_time:.2f}s)",
                    state="complete",
                    expanded=False,
                )

        

        # Display the answer and timing details in the chat interface.
        st.markdown(answer)

        # Package each retrieved source into metadata for the expandable inspector and chat history.
        source_records = [
            {
                "file": os.path.basename(d.metadata.get("source", "Unknown PDF")),
                "page": d.metadata.get("page", 0) + 1,
                "content": d.page_content,
            }
            for d in docs
        ]

        st.caption(
            f"⏱️ **Total:** {total_time:.2f}s "
            f"(Retrieval/Agent: {ret_time:.2f}s | LLM Generation: {gen_time:.2f}s) "
            f"| **Gate Passed:** `{gate_passed}`"
            + (
                f" | **Rewritten Query:** `'{rewritten_query_used}'`"
                if rewritten_query_used
                else ""
            )
        )

        with st.expander(f"🔍 Inspect Retrieved Source Chunks ({len(source_records)} chunks)"):
            for idx, src in enumerate(source_records, 1):
                st.markdown(f"**Chunk {idx} — `{src['file']}` (Page {src['page']})**")
                st.text(src["content"])
                st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "metadata": {
                    "total_time": total_time,
                    "ret_time": ret_time,
                    "gen_time": gen_time,
                    "gate_passed": gate_passed,
                    "rewritten_query": rewritten_query_used,
                    "sources": source_records,
                },
            }
        )
