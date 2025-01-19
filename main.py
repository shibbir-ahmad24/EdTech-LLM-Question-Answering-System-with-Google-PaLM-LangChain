import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Codebasics Q&A 🌱")

# Button to create the knowledge base
if st.button("Create Knowledgebase"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

# Input for the question
question = st.text_input("Ask a question:")

# Display the answer
if question:
    chain = get_qa_chain()
    response = chain({"query": question})

    st.header("Answer")
    st.write(response["result"])