from chromadb.types import Collection

from core.base_service import BaseService


class DiseaseAvgEmbeddingService(BaseService):
    """
        upsert averaged embeddings from hp_embeddings (cached dict from ont_hp collection) that are connected to the
        relevant disease from disease_to_hps (cached dict from hpoa) into the disease_avg_embeddings_collection that
        contains disease and the average embeddings of the correlating hp terms
    """

    def process_data(self) -> Collection:
        if not self.disease_to_hps:
            raise ValueError("disease to hps data is not initialized")
        if not self.disease_avg_embeddings_collection:
            raise ValueError("disease_avg_embeddings collection is not initialized")
        for disease, hps in self.disease_to_hps.items():
            average_embedding = self.data_processor.calculate_average_embedding(hps, self.hp_embeddings)
            self.disease_avg_embeddings_collection.upsert(ids=[disease], embeddings=[average_embedding.tolist()],
                                                          metadatas=[{"type": "disease"}])
        return self.disease_avg_embeddings_collection
