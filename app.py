import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from googletrans import Translator

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def translate_to_english(text, target_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text


def translate_to_user_language(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    if len(text_chunks) > 0:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Done")
    else:
        st.warning("Please upload the file")


def get_conversational_chain():
    prompt_template = """
Provide a detailed answer to the question based on the given context. If the answer is not in the provided context, indicate that it's not available.\n\n
Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, target_language):
    # Translate user question to English
    user_question_english = translate_to_english(user_question)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question_english)

    chain = get_conversational_chain()

    # Get the model response in English
    response_english = chain.invoke(
        {"input_documents": docs, "question": user_question_english},
        return_only_outputs=True
    )["output_text"]

    # Translate the model response to the user's selected language
    response_user_language = translate_to_user_language(response_english, target_language)

    st.write("Reply : ", response_user_language)


def main():
    st.set_page_config("Chat PDF")
    st.markdown(
        """
        <style>
            body {
                background-color: yellow;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.header("Chat with PDF using Gemini")
    target_language = st.selectbox("Select Your Preferred Language", ["en", "es", "fr", "de", "ja", "ko", "zh-CN"])
    user_question = st.text_input("Ask a Question from the PDF Files")

    

    if user_question:
        user_input(user_question, target_language)

    with st.sidebar:
        st.title("File Section:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)


if __name__ == "__main__":
    main()
