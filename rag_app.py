import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from langchain.chains import RetrievalQA

# -----------------------
# Load environment
# -----------------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# -----------------------
# 1. READ TXT FILE
# -----------------------
loader = TextLoader("data/hr_policy.txt", encoding="utf-8")
docs = loader.load()

# -----------------------
# 2. SPLIT TEXT INTO CHUNKS
# -----------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# -----------------------
# 3. EMBEDDINGS (MISTRAL)
# -----------------------
embeddings = MistralAIEmbeddings(
    api_key=MISTRAL_API_KEY
)

# -----------------------
# 4. CHROMA VECTOR DB
# -----------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------
# 5. LLM (MISTRAL)
# -----------------------
llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-large-latest",
    temperature=0
)

# -----------------------
# 6. RAG CHAIN
# -----------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# -----------------------
# 7. CHAT LOOP
# -----------------------
while True:
    query = input("\nAsk HR question (type exit): ")

    if query.lower() == "exit":
        break

    response = qa_chain.invoke({"query": query})
    print("\nAnswer:\n", response["result"])