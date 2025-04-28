import streamlit as st
import time
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversation_chain


def user_questions(user_input):
    response = st.session_state.conversation({"question": user_input})
    st.session_state.conversation_history = response["chat_history"]
    for i, message in enumerate(st.session_state.conversation_history):
        if i % 2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Reply: ",message.content) 


def main():
    st.set_page_config("Gen AI Information Retrival")
    st.header("Information Retrival System")

    user_input = st.text_input("Ask a question about your documents:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = None
    if user_input:
        user_questions(user_input)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on the 'Submit & Process' button", type='pdf', accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)

                st.success("PDF files uploaded and processed successfully.")

   

if __name__ == '__main__':
    main()
