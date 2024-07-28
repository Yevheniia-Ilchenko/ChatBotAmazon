import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


USER_AGENT = os.environ.get('USER_AGENT')

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    splitDocs = splitter.split_documents(docs)
    return splitDocs


def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}
    """)

    chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    retriever = vectorStore.as_retriever()

    retriever_chain = create_retrieval_chain(retriever, chain)

    return retriever_chain


docs = get_documents_from_web("https://www.amazon.co.uk/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)


# response = chain.invoke({
#     "input": "How many days can I see a refund on my bank account or credit card statement?",
# })
#
# print(response["answer"])
st.title("ChatBot for Amazon")

if 'greeted' not in st.session_state:
    st.session_state.greeted = False

if not st.session_state.greeted:
    st.session_state.greeted = [
        {"role": "system",
         "content": "Hi! I can help you with questions about the Amazon return policy. How can I help you today?"}
    ]

for msg in st.session_state.greeted:
    st.write(f"{msg['role'].capitalize()}: {msg['content']}")

user_input = st.text_input("You: ", key="input")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


if st.button("Send"):
    if user_input:
        response = chain.invoke({
            "input": user_input
        })
        st.session_state.chat_history.append((user_input, response["answer"]))

if st.session_state.chat_history:
    for question, answer in st.session_state.chat_history:
        st.write(f"Q: {question}")
        st.write(f"A: {answer}")
