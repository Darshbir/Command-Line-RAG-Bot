import os
import fitz
import argparse
import nltk
import pdfplumber
import spacy
import warnings
from nltk.corpus import stopwords
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load environment variables for API keys
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google GenAI
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.2,
    convert_system_message_to_human=True,
)

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# Text Preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    processed_words = [
        token.lemma_ for token in doc if token.text.lower() not in stop_words
    ]
    return " ".join(processed_words).strip()

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text_with_page_numbers = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text_with_page_numbers.append(f"Page {page_num}:\n{page_text}")
    return text_with_page_numbers

# Command-Line Interface Functionality for vector store setup
def setup_vector_store(texts):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )
    return Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

# Define Prompt Template for QA
template = """Use the following context to answer the question at the end. Look into the full context carefully and don't make up answers. Always end your response with "thanks for asking!".
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Function to set up QA Chain
def get_qa_chain(vector_index, question):
    return RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

def main():
    parser = argparse.ArgumentParser(description="Legal Contract Question Answering Tool")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("question", help="Question to ask about the PDF content")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

    print("Extracting text from the PDF...")
    pdf_text_with_page_numbers = extract_text_from_pdf(args.pdf_path)

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    # Add page number as metadata for each chunk
    texts = []
    for page_text in pdf_text_with_page_numbers:
        page_number = page_text.split("\n")[0].split()[1] 
        page_content = page_text[len(f"Page {page_number}:\n"):]  
        chunks = text_splitter.split_text(page_content)
        for chunk in chunks:
            texts.append((chunk, page_number))

    # Setup vector store
    print("Creating vector store for retrieval...")
    vector_index = setup_vector_store([chunk[0] for chunk in texts])

    # Get QA chain and answer
    print("Retrieving answer...")
    qa_chain = get_qa_chain(vector_index, args.question)
    result = qa_chain.invoke({"query": args.question})

    # Display answer and citations
    print("\nAnswer:")
    print(result["result"])

    # Display citations and source text with page numbers
    print("\nCitations:")
    for doc in result["source_documents"]:
        for chunk, page_number in texts:
            if chunk == doc.page_content:
                print(f"- Page {page_number}: {chunk[:300]}...")  # Show first 300 characters of chunk

if __name__ == "__main__":
    main()
