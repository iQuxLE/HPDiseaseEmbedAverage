from typing import Dict

from core.base_service import BaseService
from core.chromadb_manager import ChromaDBManager
from core.data_processor import DataProcessor
from chromadb.types import Collection


class HPEmbeddingService(BaseService):
    def process_data(self) -> Collection:
        """
            upsert hps and embeddings into hp_embeddings collection created by chromadbmanager
        """
        if not self.hp_embeddings:
            raise ValueError("HP embeddings data is not initialized")
        if not self.hp_embeddings_collection:
            raise ValueError("HP embeddings collection is not initialized")
        if self.hp_embeddings_collection is not None:
            for hp_id, data in self.hp_embeddings.items():
                embedding_list = data['embeddings']
                self.hp_embeddings_collection.upsert(ids=[hp_id], embeddings=[embedding_list],
                                                     metadatas=[{"type": "HP"}])
                # return self?
        return self.hp_embeddings_collection
