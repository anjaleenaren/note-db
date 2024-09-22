from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.indices.keyword_table import SimpleKeywordTableIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from dotenv import load_dotenv
import openai
import os
import pickle
import tempfile

class AI_TA:

    @classmethod
    def deserialize(cls, serialized):
        return pickle.loads(serialized)

    def __init__(self, class_name):
        self.class_name = class_name
        self.node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)
        self.vector_index = VectorStoreIndex([])
        self.query_engine = self.vector_index.as_query_engine()
        self.train_history = []
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=SimpleVectorStore(),
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore(),
        )
        self.vector_index = VectorStoreIndex([], storage_context=self.storage_context)
        self.keyword_index = SimpleKeywordTableIndex([], storage_context=self.storage_context)

    def serialize(self):
        return pickle.dumps(self)

    def train(self, content, files=None):
        nodes = self.node_parser.get_nodes_from_documents([Document(text=content)])
        self.vector_index.insert_nodes(nodes)
        self.train_history.append(content)
        print(f"Training complete with materials: {content[:100]}...")
        
        document = Document(text=content)
        self.vector_index.insert(document)
        self.keyword_index.insert(document)
        
        # if files:
        #     file_documents = []
        #     with tempfile.TemporaryDirectory() as temp_dir:
        #         for file in files:
        #             temp_file_path = os.path.join(temp_dir, file.name)
        #             with open(temp_file_path, 'wb') as temp_file:
        #                 temp_file.write(file.getvalue())
        #             file_documents.extend(SimpleDirectoryReader(input_files=[temp_file_path]).load_data())
            
        #     for doc in file_documents:
        #         self.vector_index.insert(doc)
        #         self.keyword_index.insert(doc)

    def query(self, query):
        vector_query_engine = self.vector_index.as_query_engine()
        keyword_query_engine = self.keyword_index.as_query_engine()
        
        query_engine_tools = [
            QueryEngineTool.from_defaults(
                query_engine=vector_query_engine,
                description="Vector-based search for semantic understanding"
            ),
            QueryEngineTool.from_defaults(
                query_engine=keyword_query_engine,
                description="Keyword-based search for specific terms"
            )
        ]
        
        query_engine = RouterQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            select_multi=True,
            verbose=True
        )
        
        response = query_engine.query(query)
        return response

    def get_history(self):
        return self.train_history
    
    
    
    