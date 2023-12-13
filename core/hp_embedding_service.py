from typing import Dict

from chromadb_manager import ChromaDBManager
from data_processor import DataProcessor
from chromadb.types import Collection


class HPEmbeddingService(ChromaDBManager, DataProcessor):
    def upsert_hps_and_embeddings(self, hp_embeddings_collection: Collection) -> Collection:
        if hp_embeddings_collection is not None:
            for hp_id, data in self.hp_embeddings.items():
                embedding_list = data['embeddings']
                hp_embeddings_collection.upsert(ids=[hp_id], embeddings=[embedding_list], metadatas=[{"type": "HP"}])
        return hp_embeddings_collection
