from llama_index.core import VectorStoreIndex, Document
from dotenv import load_dotenv
import openai
import os
import pickle
import uuid

class AI_TA:

    @classmethod
    def deserialize(cls, serialized):
        return pickle.loads(serialized)

    def __init__(self, class_name):
        self.class_name = class_name
        self.index = VectorStoreIndex([])
        self.query_engine = self.index.as_query_engine()
        self.train_history = []
    
    def serialize(self):
        return pickle.dumps(self)

    def add_doc(self, content, title, id=None):
        new_document = Document(text=content, metadata={"title": title, "id": id})
        self.index.insert(new_document)
        self.train_history.append(title)
        print(f"Training complete with materials: {title}")
        return id

    def update_doc(self, content, title, id):
        self.delete_doc(id)
        updated_document = Document(text=content, metadata={"title": title, "id": id})
        self.index.insert(updated_document)
        print(f"Updating complete with materials: {title}")

    def delete_doc(self, id):
        deleted = False
        for doc_id, doc in list(self.index.docstore.docs.items()):
            if doc.metadata.get('id') == id:
                self.index.delete(doc_id)
                deleted = True
                print(f"Deleted document with id {id}")
        
        if deleted:
            # Rebuild the index to ensure all components are updated
            # self.index = self.index.refresh()
            self.query_engine = self.index.as_query_engine()
        else:
            print(f"No document found with id {id}")

    def query(self, query):
        self.query_engine = self.index.as_query_engine()
        response = self.query_engine.query(query)
        print(f"Receive Query: {query}")
        return response

    def get_history(self):
        return self.train_history

    def get_documents(self):
        all_docs = self.index.docstore.docs
        documents = []
        for doc_id, doc in all_docs.items():
            documents.append({
                'id': doc.metadata.get('id'),
                'title': doc.metadata.get('title'),
                'content': doc.text[:100] + '...' if len(doc.text) > 100 else doc.text
            })
        return documents
