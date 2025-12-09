import os
os.environ["OPENAI_API_KEY"] = "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"

import streamlit as st
import os
from pathlib import Path
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="PDF Chatbot (RAG + FAISS + OpenAI)", layout="wide")
st.title(" PDF Chatbot (FAISS + OpenAI RAG)")
st.markdown("Upload one or more PDF documents and chat with their content.")

# ---------- PDF Upload ----------
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
process_clicked = st.button("Process Documents")

if process_clicked and uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        with BytesIO(uploaded_file.read()) as f:
            temp_path = Path("uploaded_" + uploaded_file.name)
            with open(temp_path, "wb") as out_f:
                out_f.write(f.read())
            loader = PyPDFLoader(str(temp_path))
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)
            os.remove(temp_path)

    if not all_docs:
        st.warning("No pages could be read from the PDFs.")
    else:
        st.write(f" Loaded {len(all_docs)} pages from uploaded PDFs.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        st.write(f" Split into {len(chunks)} chunks.")

        st.session_state["embedding_model"] = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state["vectorstore"] = FAISS.from_documents(chunks, st.session_state["embedding_model"])
        st.session_state["retriever"] = st.session_state["vectorstore"].as_retriever(search_type="similarity", search_kwargs={"k": 3})
        st.success("FAISS index created successfully!")

# ---------- Chat Interface ----------
if "retriever" in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Use ONLY the provided CONTEXT to answer the question clearly.
    If the answer cannot be found in the context, say:
    "I don't have enough information in the provided documents."

    CONTEXT:
    {context}

    QUESTION:
    {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(st.session_state["retriever"], document_chain)

    st.divider()
    st.subheader(" Ask a Question")
    user_question = st.text_input("Enter your question:")
    if st.button("Ask") and user_question.strip():
        with st.spinner("Retrieving and generating answer..."):
            result = rag_chain.invoke({"input": user_question})
            answer = result.get("answer", "No response generated.")
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Top Retrieved Chunks")
            docs = st.session_state["retriever"].get_relevant_documents(user_question)
            for i, d in enumerate(docs, 1):
                st.markdown(f"**{i}.** {d.page_content[:500]}...")
                st.caption(str(d.metadata))
else:
    st.info(" Upload PDFs and click 'Process Documents' to start chatting.")

