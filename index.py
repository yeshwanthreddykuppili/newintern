import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
class RAGEngine:
    def __init__(self, pdf_path: str):
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not set in environment.")
        self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = self.build_vectorstore(pdf_path)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10
            }
        )
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
        prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant helping users with claim-related queries.
Use ONLY the provided context to answer.
Rules:
- Answer clearly and conversationally.
- If exact answer exists → give it.
- If partial information exists → summarize and explain clearly.                                               
- If related info exists → explain what the document says.
- Only say "Information not available in the provided document"
  if absolutely no relevant content is found.
- Keep answers structured in bullet points.
- Do not use markdown or bold text.
- put each point on a new line.
- If more contents are present then answer them in multiple paragraphs with each paragraph containing 3-4 points.                                                                                                    
- Please do not answer in paragraphs, use bullet points for each point and each point on a new line.
- Detect the language of the user's question.
- Respond ONLY in the SAME language as the question                                                                                                   
Context:
{context}
User Question:
{input}
Answer:
""")
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)

        self.qa_chain = create_retrieval_chain(
            self.retriever,
            combine_docs_chain
        )
    def build_vectorstore(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=90
        )
        split_docs = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        return vectorstore
    def handle_general_query(self, query: str):
        greetings = ["hello", "hi", "hey"]
        if query.lower().strip() in greetings:
            return "Hello! How can I assist you with PM-JAY claim guidelines today?"
        if "who are you" in query.lower():
            return "I am your assistant helping you understand PM-JAY claim guidelines and related documents."
        return None
    def answer(self, query: str):
        general_response = self.handle_general_query(query)
        if general_response:
            return general_response
        response = self.qa_chain.invoke({"input": query})
        return response["answer"]