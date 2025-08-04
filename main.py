'''
    @ Author: Tobias Chen
    @ Date: 2025.08.02
    @ Description: Base on RAG System, we can build a knowledge base for contract information. 
'''

from fastapi import FastAPI, File, UploadFile, HTTPException,Form, Response
from typing import List
from pydantic import BaseModel
import os
import pdfplumber
import numpy as np
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import AutoTokenizer,AutoModel
import torch
from qdrant_client.models import Distance, VectorParams, PointStruct
from auto_gptq import AutoGPTQForCausalLM

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# initialize bge embedding model
embedding_model_name = "/mnt/f/ubuntu/deployment/model/bge-base-en-v1___5"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name,trust_remote_code=True,local_files_only=True)
embedding_model = AutoModel.from_pretrained(embedding_model_name)
max_chunk_size = 1024
collection_name="user_docs"
client = QdrantClient(":memory:",
                      # location = "./qdrant_data" # 持久化到磁盘
                      # 连接到远程服务器
                      # host = "http://192.168.31.13",
                      # port = 6333,
                      # api_key = "your_api_key" # 连接到远程服务器时需要提供api_key
                      )


# initialize llm model Qwen2.5
llm = AutoGPTQForCausalLM.from_pretrained(
            "/mnt/f/ubuntu/deployment/model/Qwen2___5-1___5B-Instruct-GPTQ-Int4",
            device_map="auto",
            trust_remote_code=True, 
            local_files_only=True, 
            quantize_config='gptq', 
        ).to('cuda:0')
############################################
# @ function: upload file
# @ input: file
# @ output: list[string]
# @ description: upload file to local directory and return file text content as list of string
############################################
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...),user_item= Form(...)):
    allowed_types=["application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file provided, Allowed types are: {', '.join(allowed_types)}")
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
    
        # extract text from pdf
        file_content = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                file_content+=page.extract_text().split("\n")

        file_content = [chunk for chunk in file_content if len(chunk) > 0]
        file_content = [chunk for chunk in file_content if len(chunk) <= max_chunk_size]
        # save file content to qdrant
        if is_collection_exists(collection_name=collection_name):
            # 向量化
            for text in file_content:
                embed = get_embedding(text=text)
            points = []
    
            for i, chunks in enumerate(file_content):
                doc_id = i
                points.append(
                PointStruct(
                    id=doc_id,
                    vector = get_embedding(chunks),
                    payload = {"text":chunks[0]}
                ))
            client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )   

        return  {
            "code":"200",
            "message": "File uploaded successfully",
            "file_name": file.filename,
            "user_item": user_item
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while uploading file: {str(e)}")
    finally:
        await file.close()

############################################
# @ function: search vector
# @ input: query
# @ output: list[dict]
# @ description: search documents by query and return top k documents
############################################

class RagInputParams(BaseModel):
    query: str = None
    top_k: int = 10
    temperature: float = 0.0

@app.post("/search/")
async def search_vector(rag_input_params: RagInputParams):
    query = rag_input_params.query
    top_k = rag_input_params.top_k
    temperature = rag_input_params.temperature
    results = search_from_embedding(query, top_k = top_k)
    answers = generate_doc(query,results)

    return {
        'code':200,
        'query':query,
        'answers':answers
    }




###########################################
# @ function: generate document
# @ input: query, docs
# @ output: string
# @ description: generate document by query and related documents
##################################################
def generate_doc(query: str, docs: list) -> str:
    prompt = f'''
    you are a general chat bot, you should answer the questions according to the information below, if you don't know the answer, please say "i don't know":

    customer query: {query},

    related documents:{",".join(docs)}
    '''
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    outputs = llm.generate(**inputs,max_new_tokens=200)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)








############################################
# @ function: embedding & save to qdrant
# @ input: list[string]
# @ output: bool
# @ description: embedding file content and save to qdrant
############################################
def embedding_and_save_to_qdrant(file_content: List[str], collection_name: str, vector_size: int):
    pass

############################################
# @ function: if collection exists
# @ input: collection_name
# @ output: bool
# @ description: if collection exists, if not, create one.
############################################
def is_collection_exists(collection_name: str):
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768, 
                distance=models.Distance.COSINE,  
            ),
        )
        return True
    return False

############################################
# @ function: transfer text to embedding
# @ input: text
# @ output: embedding
# @ description: transfer text to embedding using bge model
#############################################
def get_embedding(text: str) -> list:
    inputs = tokenizer(text, padding=True, return_tensors="pt", truncation=True, max_length=1024)
    # get [CLS] position vector as sentence representation
    outputs = embedding_model(**inputs)
    # 取 [CLS] 位置的向量作为句子表示
    embedding = outputs.last_hidden_state[:, 0, :].mean(dim=0).tolist()
    return embedding


############################################
# @ function: search documents by embedding
# @ input: query, top_k
# @ output: list[dict]  
# @ description: search documents by embedding and return top k documents
############################################
def search_from_embedding(query: str, top_k: int = 10) -> list:
    # search embedding in vector store
    query_embedding = get_embedding(query)
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    return [dict(doc)["payload"]['text'] for doc in results]




if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",port=8001,reload=True)