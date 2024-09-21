from llama_index.core import VectorStoreIndex, Document
from dotenv import load_dotenv
import openai
import os
import pickle

class AI_TA:

    @classmethod
    def deserialize(cls, serialized):
        return pickle.loads(serialized)

    def __init__(self, class_name):
        self.class_name = class_name
        self.index = VectorStoreIndex([])
        self.query_engine = self.index.as_query_engine()
        self.train_history=[]
    
    def serialize(self):
        return pickle.dumps(self)

    def train(self, content):
        new_document = Document(text=content)
        self.index.insert(new_document)
        self.train_history.append(content)
        print(f"Training complete with materials: {content}")

    def query(self, query):
        self.query_engine = self.index.as_query_engine()
        response = self.query_engine.query(query)
        print(f"Receive Query: {query}")
        return response

    def get_history(self):
        return self.train_history
    
    
    
    