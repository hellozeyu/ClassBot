import streamlit as st
import langchain
import time
import os
import pandas as pd
import qdrant_client
import s3
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Pinecone, Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from tzlocal import get_localzone
from utils import sizeof_fmt, dataframe_with_selections
from qdrant_client.http import models
from langchain.llms import HuggingFaceHub


def get_pdf_chunks(pdf_doc):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    loader = PyPDFLoader(pdf_doc)
    chunks = loader.load_and_split(text_splitter)
    return [chunk.page_content for chunk in chunks]


def get_vector_store():
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    st.session_state.client = client

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings,
    )

    return vector_store


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.session_state["prev_q"] = user_question

    if "chat_history" in st.session_state:
        for i, message in enumerate(st.session_state.chat_history[::-1]):
            if i % 2 == 0:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)


def clear_text():
    st.session_state["text"] = ""
    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
    st.session_state.chat_history = None
    st.session_state["prev_q"] = ""


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with AI study buddy",
                       page_icon=":books:",
                       layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        st.session_state["prev_q"] = ""

    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    if "conversation" not in st.session_state:
        # st.session_state.conversation = None
        # docsearch = Pinecone.from_existing_index(index_name, embeddings)
        st.session_state.vectorstore = get_vector_store()
        # time.sleep(1)
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

    # col1, col2 = st.columns(2)
    col1, padding, col2 = st.columns((10, 2, 10))

    with st.sidebar:
        st.subheader("Your documents")
        pdf_doc = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            type=["pdf"],
            accept_multiple_files=False,
            key=st.session_state["file_uploader_key"])
        if st.button("Process", on_click=clear_text):
            with st.spinner("Processing"):
                name = pdf_doc.name
                file_url = s3.upload(pdf_doc, pdf_doc.name)
                # get the document chunks
                pdf_chunks = get_pdf_chunks(file_url)

                # create vector store
                st.session_state.vectorstore.add_texts(pdf_chunks, metadatas=[{"key": name}]*len(pdf_chunks))
                # st.session_state.vectorstore.add_documents(pdf_chunks)
                # st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                # st.session_state.chat_history = None
            st.session_state["file_uploader_key"] += 1
            st.experimental_rerun()

    with col1:
        st.header("Chat with your AI study buddy :books:")
        user_question = st.text_input("Ask a question about the course:", key="text")
        if user_question and user_question != st.session_state["prev_q"]:
            handle_userinput(user_question)

    with col2:
        st.header("Uploaded files :file_cabinet:")
        files = s3.list_files()
        if files:
            tz = get_localzone()
            # for file in files:
            #     file["LastModified"] = file["LastModified"].astimezone(tz)
            #     file["Size"] = sizeof_fmt(file["Size"])
            #     file["Name"] = file["Key"]

            df = pd.DataFrame(files)
            df = df[["Key", "LastModified", "Size"]]
            # df['LastModified'] = df['LastModified'].dt.tz_convert(tz)
            # df["LastModified"] = df["LastModified"].map(lambda x: x.astimezone(tz))
            df["LastModified"] = pd.to_datetime(df["LastModified"])
            df["Size"] = df["Size"].map(lambda x: sizeof_fmt(x))
            st.session_state.selected_files = dataframe_with_selections(df)

            if st.button("Delete", on_click=clear_text) and st.session_state.selected_files is not None:
                for file in st.session_state.selected_files["Key"]:
                    results = st.session_state.client.scroll(
                            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
                            scroll_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="metadata.key",
                                        match=models.MatchValue(value=file),
                                    )
                                ]
                            ),
                        )
                    if results:
                        ids = [result.id for result in results[0]]
                        st.session_state.vectorstore.delete(ids)
                    s3.delete(file)
                    st.experimental_rerun()


if __name__ == '__main__':
    main()
