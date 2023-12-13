import json
import numpy as np
import logging
import chromadb

from chromadb.types import Collection
from typing import Dict, List, Optional
from utils.similarity_measures import SimilarityMeasures
from config.config_loader import load_config
from OMIMHPOExtractor import OMIMHPOExtractor
from data_processor import DataProcessor

logger = logging.getLogger(__name__)


class ChromaDBManager(DataProcessor):
    def __init__(self, path: str = None):
        if path is None:
            config = self.load_config()
            path = config['chroma_db_path']
        self.client = chromadb.PersistentClient(path=path)
        self.ont_hp = self.get_collection("ont_hp")
        self.hpoa = self.get_collection("hpoa")
        self.hp_embeddings_collection = self.create_collection("HPtoEmbeddings")
        self.disease_avg_embeddings_collection = self.create_collection("DiseaseAvgEmbeddings")
        if self.client and self.hpoa and self.ont_hp and self.hp_embeddings_collection and self.disease_avg_embeddings_collection is not None:
            logger.debug(f"{self.client} has been created and {self.hpoa} and {self.ont_hp} are available")
            logger.info(f"Creating cached Dictionaries from {self.ont_hp} and {self.hpoa}")
            self.hp_embeddings = DataProcessor.create_hpo_id_to_embedding(self.ont_hp)
            logger.info(f"{self.hp_embeddings} has been created")
            self.disease_to_hps = DataProcessor.create_disease_to_hps_dict(self.hpoa)
            logger.info(f"{self.disease_to_hps}has been created")


    @staticmethod
    def load_config():
        return load_config("../config.yaml")

    def create_collection(self, name: str, similarity: SimilarityMeasures):
        try:
            collection = self.client.create_collection(name=name, metadata={"hnsw:space": similarity})
            return collection
        except chromadb.db.base.UniqueConstraintError:
            logger.info(f"Collection {name} already exists")
            return None

    def get_collection(self, name):
        try:
            return self.client.get_collection(name)
        except Exception as e:
            logger.info(f"Error getting collection {name}: {str(e)}")
            return None

    def list_collections(self):
        return self.client.list_collections()


















    #     ####
    # # use to compare the averaged vector from one disease and do cosine similarity with the vectors of the HPs that all belong to that disease
    # @staticmethod
    # def cosine_similarity(vec1, vec2):
    #     dot_product = np.dot(vec1, vec2)
    #     magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    #     if magnitude == 0:
    #         return 0
    #     return dot_product / magnitude
    #
    # @staticmethod
    # def get_record_by_hpo_id_faster(hpo_id, hpo_id_to_data_dict) -> Optional[Dict]:
    #     """
    #     Retrieve a record by its HPO ID using a pre-compiled dictionary.
    #
    #     :param hpo_id: HPO ID of the record to retrieve.
    #     :param hpo_id_to_data_dict: A dictionary mapping HPO IDs to their data.
    #     :return: The record corresponding to the HPO ID, or None if not found.
    #     """
    #     record = hpo_id_to_data_dict.get(hpo_id)
    #     if record:
    #         return {"metadata": {"label": record["label"], "original_id": hpo_id}, "embeddings": record["embeddings"]}
    #     return None
    #
    #
    # def get_embeddings_by_hpo_ids_faster(self, ids: List[str], hpo_id_to_data_dict) -> Dict[str, Optional[np.ndarray]]:
    #     """
    #     Retrieve embeddings for a list of HPO IDs using a pre-compiled dictionary.
    #
    #     :param ids: List of HPO IDs.
    #     :param hpo_id_to_data_dict: A dictionary mapping HPO IDs to their data.
    #     :return: A dictionary mapping HPO IDs to their embeddings.
    #     """
    #     embeddings_dict = {}
    #     for hpo_id in ids:
    #         record = self.get_record_by_hpo_id_faster(hpo_id, hpo_id_to_data_dict)
    #         embeddings_dict[hpo_id] = record['embeddings'] if record else None
    #     return embeddings_dict
    #
    #
    # @staticmethod
    # # deprecated cause dict structure
    # def calculate_average_embedding(hps, embeddings_dict):
    #     embeddings = [embeddings_dict[hp_id] for hp_id in hps if hp_id in embeddings_dict]
    #     return np.mean(embeddings, axis=0) if embeddings else []
    #
    #
    # # leave in case label should be included
    # @staticmethod
    # def create_hpo_id_to_data_dict_with_embedding(collection) -> Dict[str, Dict]:
    #     """
    #     Create a dictionary mapping HPO IDs to their labels and embeddings.
    #
    #     :param collection: The collection to process.
    #     :return: A dictionary mapping HPO IDs to a dictionary of their label and embeddings.
    #     """
    #     hpo_id_to_data = {}
    #     results = collection.get(where={}, include=["metadatas", "embeddings"])
    #     for metadata, embedding in zip(results.get("metadatas", []), results.get("embeddings", [])):
    #         metadata_json = json.loads(metadata['_json'])
    #         hpo_id = metadata_json.get("original_id")
    #         label = metadata_json.get("label")
    #         if hpo_id and label:
    #             hpo_id_to_data[hpo_id] = {"label": label,
    #                                       "embeddings": embedding}  # {'HP:0005872': {'Brachytelomesophalangy' :[1,2,3, ...]}}
    #     return hpo_id_to_data
