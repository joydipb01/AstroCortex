import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Dataset
def load_documents(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".txt"):
                loader = TextLoader(path)
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(path)
            elif file.endswith(".jpg") or file.endswith(".png"):
                text = extract_text_from_image(path)
                loader = TextLoader(path)
                loader.documents = [text]
            else:
                continue
            documents.extend(loader.load())
    return documents

article_directory = "C:/Users/sumanth/Projects/AstroCortex/new_articles" # Replace with the dir path of your choice
documents = load_documents(article_directory)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

persist_directory = "db"

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding_model,
                                 persist_directory=persist_directory)

# Persist the database
vectordb.persist()
vectordb = None

# Load the database
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding_model)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
