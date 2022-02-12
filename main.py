from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd 
import torch
from pathlib import Path

from transformers import (
    CamembertTokenizer,
    Trainer
)
from thai2transformers.preprocess import process_transformers

from pydantic import BaseModel

class Text(BaseModel):
    text: str
    class Config:
        schema_extra = {
            "example": {
                "text": "แมวเธอน่ารักจังเลยอะ!"
            }
        }

class Dataset(torch.utils.data.Dataset):    
    def __init__(self, encodings, labels=None):          
        self.encodings = encodings        
        self.labels = labels
     
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.on_event("startup")
def load_model():
    global Testmodel, tokenizer
    Model_Path = "model.pkl"
    model = torch.load(Model_Path, map_location=torch.device('cpu'))
    Testmodel = Trainer(model)
    tokenizer = CamembertTokenizer.from_pretrained(
                                  'airesearch/wangchanberta-base-att-spm-uncased',
                                  revision='main')
    tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', '<_>']

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API :)'}


@app.post('/predict')
def get_sentiment(data: Text):
    print(data)
    received = data.dict()
    texts = received['text']

    data = tokenizer([texts], padding=True, truncation=True, max_length=512)
    data = Dataset(data)
    raw_pred, _, _ = Testmodel.predict(data) 

    pred = np.argmax(raw_pred, axis=1)

    if pred == 0:
      result = 'Positive'
    elif pred == 1:
      result = 'Neutral'
    elif pred == 2:
      result = 'Negative'

    return {'prediction': result}