from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from load_from_pinecone_queryLLM import *
from dotenv import load_dotenv

app = FastAPI()

#uvicorn app:app --host 0.0.0.0 --port 8000 --reload
load_dotenv()

@app.get("/hello")
async def hello():
    return "Tesying Hello World!"


@app.get("/CallLLM")
async def callLLM(query:str):

    embeddings=create_embeddings()

    #Function to pull index data from Pinecone
    index=pull_from_pinecone("","gcp-starter","chatbot",embeddings)
    
    #This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
    relavant_docs=get_similar_docs(index,query)

    #This will return the fine tuned response by LLM
    response=get_answer(relavant_docs,query)
    
    return response

@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"

