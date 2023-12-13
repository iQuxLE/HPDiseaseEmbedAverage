from typing import List
from chromadb_manager import ChromaDBManager
from data_processor import DataProcessor


class QueryService(ChromaDBManager, DataProcessor):
    def query_diseases_by_hpo_terms(self, hpo_ids: List[str]) -> List:
        """
        Queries the 'DiseaseAvgEmbeddings' collection for diseases closest to the average embeddings of given HPO terms.

        :param hpo_ids: List of HPO term IDs.
        :return: List of diseases sorted by closeness to the average HPO embeddings.
        """
        avg_embedding = self.calculate_average_embedding_from_cachedDict(hpo_ids, self.hp_embeddings)

        if avg_embedding is None:
            return "No valid embeddings found for provided HPO terms."

        query_results = self.disease_avg_embeddings_collection.query(
            query_embeddings=[avg_embedding.tolist()],
            n_results=10,
            include=["embeddings", "distances"]
        )

        disease_ids = query_results['ids'][0] if 'ids' in query_results and query_results['ids'] else []
        distances = query_results['distances'][0] if 'distances' in query_results and query_results['distances'] else []
        sorted_results = sorted(zip(disease_ids, distances), key=lambda x: x[1])

        return sorted_results
