import json
from typing import Dict

from chromadb.types import Collection
import numpy as np

from core.OMIMHPOExtractor import OMIMHPOExtractor


class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def create_hpo_id_to_embedding(collection: Collection) -> Dict:
        """
        Create a dictionary mapping HPO IDs to embeddings.

        :param collection: The collection to process
        :return: A dictionary mapping HPO IDs to a dictionary of their label and embeddings.
        """
        hpo_id_to_data = {}
        results = collection.get(include=["metadatas", "embeddings"])
        for metadata, embedding in zip(results.get("metadatas", []), results.get("embeddings", [])):
            metadata_json = json.loads(metadata['_json'])
            hpo_id = metadata_json.get("original_id")
            if hpo_id:
                hpo_id_to_data[hpo_id] = {"embeddings": embedding}  # #{'HP:0005872': [1,2,3, ...]}
        return hpo_id_to_data

    @staticmethod
    def create_disease_to_hps_dict(collection: Collection) -> Dict:
        """
        Creates a dictionary mapping diseases (OMIM IDs) to their associated HPO IDs.

        :param collection: The collection to process
        :return: Dictionary with diseases as keys and lists of corresponding HPO IDs as values.
        """
        disease_to_hps_dict = {}
        results = collection.get(where={}, include=["metadatas"])
        for item in results.get("metadatas"):
            metadata_json = json.loads(item["_json"])
            disease = metadata_json.get("disease")
            phenotype = metadata_json.get('phenotype')
            if disease and phenotype:
                if disease not in disease_to_hps_dict:
                    disease_to_hps_dict[disease] = [phenotype]
                else:
                    disease_to_hps_dict[disease].append(phenotype)
        return disease_to_hps_dict

    @staticmethod
    def extract_and_use_omim_hpo_mappings(file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        return OMIMHPOExtractor.extract_omim_hpo_mappings(data)

    @staticmethod
    def calculate_average_embedding_from_cachedDict(hps: list, embeddings_dict: Dict) -> np.ndarray:
        """
        Calculates the average embedding for a given set of HPO IDs.

        :param hps: List of HPO IDs.
        :param embeddings_dict: Dictionary mapping HPO IDs to their embeddings.
        :return: A numpy array representing the average embedding for the HPO IDs.
        """
        embeddings = [embeddings_dict[hp_id]['embeddings'] for hp_id in hps if hp_id in embeddings_dict]
        return np.mean(embeddings, axis=0) if embeddings else []
