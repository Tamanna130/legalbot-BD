import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import cohere
import time

# Load API keys
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
cohere_api_key = os.getenv("COHERE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM and Cohere
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
co = cohere.ClientV2(cohere_api_key)

st.title("Chat to Learn About The Succession Act of Bangladesh")

# =============================
# Scraping Function
# =============================
def scrape_bdlaws_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 403:
        raise Exception("Access denied (403 Forbidden)")

    soup = BeautifulSoup(response.text, 'html.parser')

    content = ""
    act_name = soup.find('p', class_ = "act-part-name").text.strip()
    content += f"{act_name}\n\n"

    rows = soup.find_all("div", class_="row")
    current_chapter_no = ""
    current_chapter_name = ""


    for row in rows:
        chapter_group = row.find("div", class_="act-chapter-group")

        if chapter_group:  # New chapter
            chapter_no = chapter_group.find("p", class_="act-chapter-no")
            chapter_name = chapter_group.find("p", class_="act-chapter-name")

            current_chapter_no = chapter_no.get_text(strip=True) if chapter_no else "No Chapter No"
            current_chapter_name = chapter_name.get_text(strip=True) if chapter_name else "No Chapter Name"

            content += f"{current_chapter_no}\n{current_chapter_name}\n\n"

        
        head = row.find("div", class_="col-sm-3 txt-head")
        details = row.find("div", class_="col-sm-9 txt-details act-part-details")

        if head and details:
            head_text = head.get_text(strip=True)
            details_text = details.get_text(strip=True)
            content += f"{head_text}:\n{details_text}\n\n"

    return [Document(page_content=content)]

# =============================
# Prompt Template
# =============================
template = """You are a legal assistant specializing in inheritance law in Bangladesh.

Answer the following question strictly based on the provided legal context. If the answer is not clearly present, respond with "The answer is not found in the given legal text."

Context:
{context}

Question:
{question}

Answer:"""

prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

# =============================
# Embed & Vectorize Documents
# =============================
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(google_api_key = google_api_key, model="models/embedding-001")
        scraped_docs = scrape_bdlaws_page("http://bdlaws.minlaw.gov.bd/act-138/part-details-152.html")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(scraped_docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user in chatbot UI
if prompt1 := st.chat_input("Ask a question about inheritance law..."):
    with st.chat_message("user"):
        st.markdown(prompt1)
    st.session_state.messages.append({"role": "user", "content": prompt1})

    if "vectors" in st.session_state:
        try:
            start = time.process_time()

            # Step 1: Retrieve from FAISS
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 10})
            retrieved_docs = retriever.get_relevant_documents(prompt1)
            doc_texts = [doc.page_content for doc in retrieved_docs]

            # Step 2: Rerank with Cohere
            rerank_response = co.rerank(
                query=prompt1,
                documents=doc_texts,
                top_n=5,
                model="rerank-v3.5"
            )

            top_docs = [doc_texts[r.index] for r in rerank_response.results]
            combined_context = "\n\n".join([doc for doc in top_docs if isinstance(doc, str)])

            # Step 3: LLM + Prompt
            chain = prompt_template | llm
            final_response = chain.invoke({"context": combined_context, "question": prompt1})
            # response_time = round(time.process_time() - start, 2)

            # Step 4: Output in chat format
            assistant_reply = f"{final_response.content}\n\n"
            with st.chat_message("assistant"):
                st.markdown(assistant_reply)
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            st.markdown("Please scrape and embed the website content first.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please scrape and embed the website content first."
        })

# Button to trigger embedding
if __name__ == "__main__":
    vector_embedding()

