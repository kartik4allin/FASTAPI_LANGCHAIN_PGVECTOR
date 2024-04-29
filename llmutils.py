from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

def get_answer(docs,user_input):
    llm = OpenAI(model_name="gpt-3.5-turbo-1106")
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_input)
    return response