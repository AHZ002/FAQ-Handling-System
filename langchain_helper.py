import os
from langchain_community.llms import GooglePalm
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from secret import GOOGLE_API_KEY as google_api_key
from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_api_key, temperature=0.1)

embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)

vectordb_file_path = "faiss_index"


def create_vector_db():
    loader = PyPDFLoader(r"C:\Users\ABDUL_HADI\Desktop\ChatBot2\Rappo qa.pdf")
    documents = loader.load()

    raw_text = ''
    for i, doc in enumerate(documents):
        text = doc.page_content
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=600,
        chunk_overlap=50,
        length_function=len,
    )

    texts = text_splitter.split_text(raw_text)
    vectordb = FAISS.from_texts(texts, embeddings)
    vectordb.save_local(vectordb_file_path)
    return vectordb


def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
    If the answer is not found in the context, kindly state "I don't know. Feel free to ask your query on admin@buildrappo.com." Don't try to make up an answer.

    If the context does not contain any information about the question, say "I don't know. Feel free to ask your query on admin@buildrappo.com."

    CONTEXT: {context}

    QUESTION: {question}

    ANSWER: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectordb.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})
    return chain


if __name__ == "__main__":
    if not os.path.exists(vectordb_file_path):
        create_vector_db()

    chain = get_qa_chain()
    # Query the chain and print the result
    queries = ["Who is ahz?", "What is viggy?"]
    for query in queries:
        result = chain({"query": query})
        print(f"Question: {query}")
        print(f"Answer: {result['result']}\n")
