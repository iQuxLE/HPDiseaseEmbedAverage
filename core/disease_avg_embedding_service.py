from chromadb.types import Collection
from chromadb_manager import ChromaDBManager


class DiseaseAvgEmbeddingService(ChromaDBManager):
    def upsert_disease_and_avg_embeddings(self, disease_avg_collection: Collection) -> Collection:
        for disease, hps in self.disease_to_hps.items():
            average_embedding = self.calculate_average_embedding_from_cachedDict(hps, self.hp_embeddings)
            disease_avg_collection.upsert(ids=[disease], embeddings=[average_embedding.tolist()],
                                          metadatas=[{"type": "disease"}])
        return disease_avg_collection
