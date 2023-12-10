import json
from typing import Dict, List, Optional

import chromadb
import numpy as np

from OMIMHPOExtractor import OMIMHPOExtractor


class ChromaDBManager:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path=path)

    def create_collection(self, name):
        try:
            collection = self.client.create_collection(name)
            return collection
        except chromadb.db.base.UniqueConstraintError:
            print(f"Collection {name} already exists")
            return None

    def get_collection(self, name):
        try:
            return self.client.get_collection(name)
        except Exception as e:
            print(f"Error getting collection {name}: {str(e)}")
            return None

    def list_collections(self):
        return self.client.list_collections()

    @staticmethod
    def extract_and_use_omim_hpo_mappings(file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        return OMIMHPOExtractor.extract_omim_hpo_mappings(data)

    @staticmethod
    def create_hpo_id_to_data_dict_with_embedding(collection) -> Dict[str, Dict]:
        """
        Create a dictionary mapping HPO IDs to their labels and embeddings.

        :param collection: The collection to process.
        :return: A dictionary mapping HPO IDs to a dictionary of their label and embeddings.
        """
        hpo_id_to_data = {}
        results = collection.get(where={}, include=["metadatas", "embeddings"])
        for metadata, embedding in zip(results.get("metadatas", []), results.get("embeddings", [])):
            metadata_json = json.loads(metadata['_json'])
            hpo_id = metadata_json.get("original_id")
            label = metadata_json.get("label")
            if hpo_id and label:
                hpo_id_to_data[hpo_id] = {"label": label,
                                          "embeddings": embedding}  # {'HP:0005872': {'Brachytelomesophalangy' :[1,2,3, ...]}}
        return hpo_id_to_data

    # use this for metric
    @staticmethod
    def create_hpo_id_to_embedding(collection) -> Dict[str, Dict]:
        """
        Create a dictionary mapping HPO IDs to embeddings.

        :param collection: The collection to process.
        :return: A dictionary mapping HPO IDs to a dictionary of their label and embeddings.
        """
        hpo_id_to_data = {}
        results = collection.get(where={}, include=["metadatas", "embeddings"])
        for metadata, embedding in zip(results.get("metadatas", []), results.get("embeddings", [])):
            metadata_json = json.loads(metadata['_json'])
            hpo_id = metadata_json.get("original_id")
            if hpo_id:
                hpo_id_to_data[hpo_id] = {"embeddings": embedding}  # #{'HP:0005872': [1,2,3, ...]}
        return hpo_id_to_data

    def get_embeddings_by_hpo_ids_faster(self, ids: List[str], hpo_id_to_data_dict) -> Dict[str, Optional[np.ndarray]]:
        """
        Retrieve embeddings for a list of HPO IDs using a pre-compiled dictionary.

        :param ids: List of HPO IDs.
        :param hpo_id_to_data_dict: A dictionary mapping HPO IDs to their data.
        :return: A dictionary mapping HPO IDs to their embeddings.
        """
        embeddings_dict = {}
        for hpo_id in ids:
            record = self.get_record_by_hpo_id_faster(hpo_id, hpo_id_to_data_dict)
            embeddings_dict[hpo_id] = record['embeddings'] if record else None
        return embeddings_dict

    @staticmethod
    def get_record_by_hpo_id_faster(hpo_id, hpo_id_to_data_dict) -> Optional[Dict]:
        """
        Retrieve a record by its HPO ID using a pre-compiled dictionary.

        :param hpo_id: HPO ID of the record to retrieve.
        :param hpo_id_to_data_dict: A dictionary mapping HPO IDs to their data.
        :return: The record corresponding to the HPO ID, or None if not found.
        """
        record = hpo_id_to_data_dict.get(hpo_id)
        if record:
            return {"metadata": {"label": record["label"], "original_id": hpo_id}, "embeddings": record["embeddings"]}
        return None

    @staticmethod
    def calculate_average_embedding_from_cachedDict(hps, embeddings_dict):
        embeddings = [embeddings_dict[hp_id]['embeddings'] for hp_id in hps if hp_id in embeddings_dict]
        return np.mean(embeddings, axis=0) if embeddings else []

    @staticmethod
    # deprecated cause dict structure
    def calculate_average_embedding(hps, embeddings_dict):
        embeddings = [embeddings_dict[hp_id] for hp_id in hps if hp_id in embeddings_dict]
        return np.mean(embeddings, axis=0) if embeddings else []

    @staticmethod
    def extract_omim_hpo_mappings_from_collection(collection):
        omim_hpo_dict = {}

        for item in collection.get():
            metadata = item['metadatas'][0]
            omim_id = metadata['disease']
            hpo_id = metadata['phenotype']

            if omim_id in omim_hpo_dict:
                omim_hpo_dict[omim_id].add(hpo_id)
            else:
                omim_hpo_dict[omim_id] = {hpo_id}

        for omim in omim_hpo_dict:
            omim_hpo_dict[omim] = list(omim_hpo_dict[omim])

        return omim_hpo_dict

    @staticmethod
    # just for testing with lim 1
    # works with dic directly instead of collection as this limit = 1 gets a dict
    def extract_omim_hpo_mappings_from_dict_via_jsonstring(record):
        omim_hpo_dict = {}

        metadata_json = record['metadatas'][0]['_json']
        metadata = json.loads(metadata_json)

        omim_id = metadata['disease']
        hpo_id = metadata['phenotype']

        if omim_id not in omim_hpo_dict:
            omim_hpo_dict[omim_id] = []
        omim_hpo_dict[omim_id].append(hpo_id)

        return omim_hpo_dict

    # use to compare the averaged vector from one disease and do cosine similarity with the vectors of the HPs that all belong to that disease
    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if magnitude == 0:
            return 0
        return dot_product / magnitude
