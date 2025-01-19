# EdTech LLM: Question Answering System with Google PaLM & LangChain

This project is an end-to-end implementation of a question-answering system using Google PaLM and LangChain for Codebasics, an e-learning platform (website: codebasics.io) that offers data-centric courses and bootcamps. The system is designed to assist learners by addressing their queries, typically asked via Discord or email, through a Streamlit-based user interface, providing accurate and efficient responses.

## **Project Highlights**

**Real Data Integration:** Utilizes a CSV file containing FAQs currently in use by CodeBasics.

**LLM-Powered Q&A System:** Combines LangChain and Google PaLM to build an advanced question-answering system, reducing reliance on human support staff.

**Interactive User Interface:** Features a Streamlit-based interface, enabling students to ask questions and receive timely, accurate responses.

**HuggingFace Embeddings:** Implements HuggingFace’s all-MiniLM-L6-v2 model for generating text embeddings.

**Efficient Vector Search:** Leverages FAISS for vector database creation and retrieval functionalities.

## **Learning Objectives**

This project provides hands-on experience with:

**LangChain and Google PaLM:** For building robust LLM-driven Q&A systems.

**Streamlit:** For developing interactive and user-friendly web applications.

**HuggingFace Embeddings:** For effective text representation using pre-trained models.

**FAISS:** For managing efficient vector databases for retrieval tasks.

## **Project Structure**

**main.py:** The primary script for running the Streamlit application.

**langchain_helper.py:** Contains the LangChain implementation and related functionalities.

**requirements.txt:** Lists all the dependencies required for the project.

**.env:** Stores the Google API key and other configuration details securely.
