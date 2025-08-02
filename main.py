'''
    @ Author: Tobias Chen
    @ Date: 2025.08.02
    @ Description: Base on RAG System, we can build a knowledge base for contract information. 
'''

from fastapi import FastAPI, File, UploadFile, HTTPException,Form
from typing import List
from pydantic import BaseModel
import os
import pdfplumber
import numpy as np
import uvicorn

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class RagInputParams(BaseModel):
    query: str
    top_k: int = 10
    temperature: float = 0.0

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
        print(user_item.user_id)
        return file_content 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while uploading file: {str(e)}")
    finally:
        await file.close()

############################################
# @ function: embedding & save to qdrant
# @ input: list[string]
# @ output: bool
# @ description: embedding file content and save to qdrant
############################################
def embedding_and_save_to_qdrant(file_content: List[str], collection_name: str, vector_size: int):
    pass


##
# @ function: search in qdrant
# @ input: query, top_k, temperature


if __name__ == '__main__':
    uvicorn.run("main:app", host="192.168.31.13",port=8000,reload=True)