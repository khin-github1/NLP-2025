import os
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set API Key for Groq
os.environ["GROQ_API_KEY"] ="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"

# Initialize Flask app
app = Flask(__name__)

# Path to save/load FAISS index
faiss_index_path = "faiss_index"

# Load documents and create retriever
def initialize_retriever():
    pdf_files = [r"D:\AIT_lecture\NLP\code\NLP-A6\CV_Khin_Yadanar_Hlaing.pdf"]
    web_links = ["https://www.linkedin.com/in/kyhlaing/"]
    documents = []

    # Load PDF documents
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            pdf_loader = PyMuPDFLoader(pdf_file)
            documents.extend(pdf_loader.load())

    # Load LinkedIn (or other web) data
    for link in web_links:
        web_loader = WebBaseLoader(link)
        documents.extend(web_loader.load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings using SentenceTransformer
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if FAISS index exists
    if os.path.exists(faiss_index_path):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(faiss_index_path)

    # Create retriever
    retriever = vectorstore.as_retriever()
    return retriever

# Initialize retriever
retriever = initialize_retriever()

# Define a custom prompt template
prompt_template = PromptTemplate(
    template="You are an AI assistant. Answer the following question based on the provided context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"]
)

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Initialize QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle chatbot responses
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = qa_chain.invoke({"query": user_input})
    
    # Extract the answer and source documents
    answer = response['result']
    sources = [doc.metadata for doc in response['source_documents']]
    
    return jsonify({
        'answer': answer,
        'sources': sources
    })

if __name__ == '__main__':
    app.run(debug=True)