## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Replaced Chroma import with FAISS import
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import random

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

selected_tab = st.sidebar.radio("Go to", ["Upload PDFs", "Chatbot"])

if selected_tab == "Upload PDFs":
    st.header("Upload PDFs")
    st.write("Upload your PDF files here.")
    if st.button('Reset'):
        st.session_state.uploaded_files.clear()
        st.session_state.documents.clear()
        st.session_state.submitted = False
        #st.write(st.session_state)
    # uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None

    if (st.session_state.uploaded_files is None) or (st.session_state.uploaded_files == []):
        # File uploader widget
        uploaded_files = st.file_uploader(
            "Choose a PDF file", type="pdf", accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

    # Submit button
    if st.session_state.uploaded_files:
        if st.button("Submit"):
            st.session_state.submitted = True

    # Processing
    if st.session_state.get("submitted", False):
        st.success("Files have been submitted! Processing...")
        documents = []
        # Now you can access the uploaded files like this:
        for uploaded_file in st.session_state.uploaded_files:
            st.write(f"Processing file: {uploaded_file.name}")
            # Add your processing logic here
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        st.session_state.documents = documents
        st.session_state.submitted = False  # Reset the submitted state


elif selected_tab == "Chatbot":
    st.title("Chatbot")
    st.write("Ask questions about the uploaded PDFs.")
    if st.button("Clear Session"):
        del st.session_state["session_id"]
        del st.session_state["store"]
        del st.session_state["messages"]
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(random.randint(10, 100))
        session_id = st.session_state.session_id
    else:
        session_id = st.session_state.session_id

    if 'store' not in st.session_state:
        st.session_state.store = {}
    documents = st.session_state.documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    
    # Replaced Chroma with FAISS for vector store
    # FAISS setup for indexing documents
    faiss_index = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = faiss_index.as_retriever()    

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Please enter your question"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": prompt},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        response = response['answer']
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    # st.write(st.session_state.store)
    # st.write("Chat History:", session_history.messages)
