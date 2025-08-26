import os
import json
import chromadb
import uuid
import app_config

class chroma_agent:
    def __init__(self, db_path="./db", collection_name="reports"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_document(self, document, metadata=None):
        document_id = str(uuid.uuid4())
        self.collection.add(
            ids=[document_id],
            documents=[document],
            metadatas=[metadata]
        )
        return document_id

    def query(self, query_text, n_results=1):
        results = self.collection.query(
            query_texts=[query_text],
            include=['distances', 'embeddings', 'documents', 'metadatas'],
            n_results=n_results
        )
        return results