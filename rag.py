# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# loader = PyPDFLoader("birds.pdf")
# raw_docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# chunks = splitter.split_documents(raw_docs)

free_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

# vector_db = Chroma.from_documents(documents=chunks,embedding=free_embeddings, persist_directory="./persist_birds")

after_pushing_to_directory = Chroma(persist_directory="./persist_birds", embedding_function=free_embeddings)


# db_content = vector_db.get(limit=1, include=['documents', 'metadatas'])

# print("--Chrome DB inspection--")
# print(f"Stored text snippet: {db_content['documents'][0][:100]}...")
# print(f"From Metadata: {db_content['metadatas'][0]}")

query = "Which birds are discussed in this documents ? "
docs = after_pushing_to_directory.similarity_search(query, k=2)

print("doc: ", docs)

for doc in docs:
    print(f"Page {doc.metadata["page"]}: {doc.page_content[:150]}")
    print("-" * 20)
