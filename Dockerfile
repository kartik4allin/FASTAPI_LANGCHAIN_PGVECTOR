FROM python:3.8

WORKDIR /FASTAPI_DOCKER

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install git -y
RUN pip3 install -r requirements.txt
#RUN pip3 install "git+https://github.com/openai/whisper.git" 
#RUN apt-get update && apt-get install -y ffmpeg

RUN pip install --upgrade pinecone-client

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
