import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

def get_documents_name(directory):
    try:
        return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    except FileNotFoundError:
        print(f"A pasta '{directory}' não foi encontrada.")
        return []
    except PermissionError:
        print(f"Permissão negada para acessar a pasta '{directory}'.")
        return []
    
def load_pdfs(pdf_paths):
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        for document in loader.load():
            documents.append(document)
    return documents

def get_documents_content(directory):
    pdfs = get_documents_name(directory)
    documents = load_pdfs(pdfs)
    return documents


if __name__ == "__main__":
    load_dotenv()

    api_key = os.environ["GOOGLE_API_KEY"]
    print(api_key)

    documents = get_documents_content("Documents")
    embending = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
    vector_db = FAISS.from_documents(documents=documents, embedding=embending)

