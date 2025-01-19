import os
import torch
import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer
from langchain.document_loaders import CSVLoader
import google.generativeai as palm
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORDB_FILE_PATH = "faiss_index"
CSV_FILE_PATH = "codebasics_faqs.csv"

# Configure Google PaLM API
def configure_palm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
    palm.configure(api_key=api_key)

# Custom PaLM LLM class
class PaLMWrapper(BaseLLM):
    def __init__(self):
        super().__init__()  # Initialize the base class
        configure_palm()  # Configure the Google PaLM API

    def _call(self, prompt: str, stop: list = None) -> str:
        # Handle stop argument
        response = palm.generate_text(prompt=prompt)
        result = response.result
        if stop and any(stop_str in result for stop_str in stop):
            return result.split(stop[0])[0]  # Stop at the first stop string
        return result

    def _agenerate(self, prompts: list, stop: list = None) -> list:
        # Async version of _call
        return [self._call(prompt, stop) for prompt in prompts]

    def _generate(self, prompts: list, stop: list = None) -> list:
        # Synchronous version of _call
        return [self._call(prompt, stop) for prompt in prompts]

    @property
    def _llm_type(self) -> str:
        return "PaLM"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": "google/palm"}  # Return a dictionary with model_name

# Load the pre-trained model and tokenizer
def load_huggingface_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Generate embeddings for text
def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

# Create FAISS vector database
def create_vector_db():
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"CSV file '{CSV_FILE_PATH}' not found.")
    
    df = pd.read_csv(CSV_FILE_PATH, encoding='latin1')
    data = [{"page_content": row['prompt'], "metadata": {"response": row['response']}} for _, row in df.iterrows()]
    texts = [item["page_content"] for item in data]

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectordb = FAISS.from_texts(texts, embeddings)
    vectordb.save_local(VECTORDB_FILE_PATH)

# Load or create FAISS vector database
def load_or_create_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    try:
        vectordb = FAISS.load_local(VECTORDB_FILE_PATH, embeddings)
    except FileNotFoundError:
        create_vector_db()
        vectordb = FAISS.load_local(VECTORDB_FILE_PATH, embeddings)
    return vectordb

# Build RetrievalQA chain
def get_qa_chain():
    vectordb = load_or_create_vector_db()
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """
    Given the following context and a question, generate an answer based on this context only.
    Use as much text as possible from the "response" section in the source document without making major changes.
    If the answer is not found in the context, state "I don't know." Do not fabricate an answer.

    CONTEXT: {context}

    QUESTION: {question}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Initialize the custom PaLM LLM wrapper
    llm = PaLMWrapper()  

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        output_key="result",  # Specify the output key
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# Main execution
if __name__ == "__main__":
    try:
        # Ensure the vector database is created
        create_vector_db()
        # Get the QA chain
        chain = get_qa_chain()
        query = "Do you have a JavaScript course?"
        response = chain({"query": query})  # Pass the input as a dictionary
        print(response)
    except Exception as e:
        print(f"Error: {e}")
