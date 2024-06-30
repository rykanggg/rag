from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader

DATA_PATH = "data"
CHROMA_PATH = "chroma"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Go to langchain and see document loaders to pick from any various file formats
# or other 3rd party documents

def load_unstruct_docs():
    loader = UnstructuredPDFLoader("data/Blackrock.pdf")
    return loader.load()

documents = load_unstruct_docs()
print(documents[0])

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

chunks = split_documents(documents)
print(chunks[0])

from langchain_community.embeddings.bedrock import BedrockEmbeddings

def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )
    return embeddings

# Or choose any embedding function listed on the langchain website

from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_local_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

# Download Ollama from official website and install with
# ollama pull llama2
# ollama serve (to serve as a REST local API)

from langchain.vectorstores.chroma import Chroma

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_local_embedding_function()
    
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:

            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, id=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
        
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
    
        chunk_id =f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

add_to_chroma(chunks)