#import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


#Function to pull index data from Pinecone
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    Pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
def get_similar_docs(index,query,k=5):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_answer(docs,user_input):
    llm = OpenAI(model_name="gpt-3.5-turbo-1106")
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_input)
    return response
