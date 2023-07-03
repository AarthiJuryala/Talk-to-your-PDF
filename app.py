import os
import openai
import nltk
import streamlit as st
import PyPDF2

nltk.download("punkt")

def split_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)

    texts = []
    for page in pdf_reader.pages:
        texts.append(page.extract_text())

    return texts

def process_query(query, texts):
    context = "\n".join(texts)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=context + "\n\nQuery: " + query,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

def app():
    st.sidebar.write("Enter the required keys to get an answer.")
    openai.api_key = st.sidebar.text_input("Enter OpenAI API key:")
    st.title("Ask Your PDF Files")
    st.write("Upload a PDF file, and enter a query to get an answer.")

    file = st.file_uploader("Upload PDF file", type=["pdf"])
    if not file:
        st.stop()

    texts = split_pdf_text(file)

    query = st.text_input("Enter a query:")

    if st.button("Execute"):
        answer = process_query(query, texts)
        st.write("Answer:")
        st.write(answer)

if __name__ == "__main__":
    app()
