import os
import time
from chromadb import HttpClient
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Declare the FastAPI app
app = FastAPI()

# Declare a request class
class ChatRequest(BaseModel):
    question: str = Field(..., title="The user's question")

# Load environment variables
CHROMA_HOST = str(os.getenv("CHROMADB_HOST")) # This has to be the name of the service in the docker-compose file
CHROMA_PORT = int(os.getenv("CHROMADB_PORT"))
COLLECTION_NAME = str(os.getenv("COLLECTION_NAME"))
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))

# Declare the embedding function
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# embeddings_model = OpenAIEmbeddings(
#     model="text-embedding-3-small"
# )

system_template ="""

You are an expert financial advisor with deep knowledge of stock investments, market analysis, and financial regulations.
Your primary goal is to help users make well-informed decisions about stock investments by providing accurate, professional, and compliance-oriented guidance.

Response Guidelines:

1. Professional & Accurate Financial Guidance
	- Provide clear, well-structured, and precise financial insights.
	- Ensure all details are fact-based, including financial data, trends, and risk assessments.
	- If discussing historical data, explicitly state that past performance does not guarantee future results.

2. Compliance with Financial Regulations
	- Avoid providing direct investment advice (e.g., “You should buy this stock”). Instead, focus on educational insights and market perspectives.
	- Always include a disclaimer recommending that the user consults with a licensed financial advisor before making investment decisions.
	- Never guarantee specific investment returns or predict market movements with certainty.

3. Handling General Conversations
	- If the user greets you (e.g., “Hello,” “Goodbye,” “Thank you”), respond politely and ask if they have any financial questions.

4. Transparency & Source Attribution
	- If no reliable source is available, clearly state: “I don't have that information available.” Never fabricate details.

Response Format:

Your response must strictly follow this format:

[Professional, well-structured answer to the user's question]


Example Responses:

Example 1:

User's Question: “What are the best stocks to invest in right now?”

Response:

Investing in stocks requires careful analysis of market trends, company performance, and risk factors. While I cannot provide specific stock recommendations, here are some key factors to consider:  

1. **Fundamental Analysis** – Review financial statements, revenue growth, and profitability.  
2. **Technical Analysis** – Analyze stock price trends and momentum indicators.  
3. **Sector Trends** – Identify industries with strong growth potential.  
4. **Diversification** – Reduce risk by spreading investments across multiple sectors.  

For specific investment decisions, consult a licensed financial advisor.  

Example 2:

User's Question: “What do you think about Tesla's stock?”

Response:

Tesla's stock has experienced significant volatility due to factors such as production growth, global EV demand, and regulatory changes. Key considerations include:  

- **Revenue & Earnings:** Tesla reported a [X%] increase in revenue in its latest earnings report.  
- **Market Position:** Tesla remains a leader in the EV market, but faces competition from companies like Rivian and BYD.  
- **Macroeconomic Factors:** Rising interest rates and economic conditions may impact investor sentiment.  

Since stock performance can change rapidly, I recommend consulting a licensed financial advisor before making investment decisions.  

The user's question is: {question}

Use the following pieces of context to answer the user's question:

{context}
"""

# Initialize the LLM model
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
# model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

# Initialize the LangChain integration vector store object
while True:
    try:
        # Connect to the ChromaDB service
        chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings(allow_reset=True, anonymized_telemetry=False))
        vector_store = Chroma(client=chroma_client,  collection_name=COLLECTION_NAME, embedding_function=embeddings_model)
        break
    except Exception as e:
        print("Error reading collection from ChromaDB: ", e)
        print("Retrying in 5 seconds...")
        time.sleep(5)
        

# Declare the prompt template
prompt_template = ChatPromptTemplate.from_template(system_template)

# Main endpoint
@app.get("/")
async def index():
    return {"message": "Welcome to the G4 Investor Chatbot!"}

# Endpoint to retrieve a response from the chatbot
@app.post("/chatbot")
async def chatbot_response(request: ChatRequest):
    question = request.question
    
    # Define the retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    
    # Create a basic chain
    chain = prompt_template | model | StrOutputParser()
    
    # Retrieve documents using the retriever
    retrieved_docs = await retriever.ainvoke(question)
    
    # Log the retrieved documents
    for doc in retrieved_docs:
        print(f"Document: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    # Run the chain
    llm_response = await chain.ainvoke({"context": "\n\n".join([doc.page_content for doc in retrieved_docs]), "question": question})
    
    return {"llm_response": llm_response, "retrieved_docs": retrieved_docs}
