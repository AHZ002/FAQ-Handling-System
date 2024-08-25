import os
import streamlit as st
from langchain_helper2 import create_vector_db, get_qa_chain

st.title("Ask Rappo")

# Button to create knowledgebase
if st.button("Create Knowledgebase"):
    # if not os.path.exists("faiss_index"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

# Input for user question
question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain.invoke(question)

    st.header("Answer")
    st.write(response["result"])
