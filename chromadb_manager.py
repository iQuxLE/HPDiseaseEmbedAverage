import json
from typing import Dict, List, Optional

import chromadb
from chromadb.types import Collection
import numpy as np

from OMIMHPOExtractor import OMIMHPOExtractor


class ChromaDBManager:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path=path)

    def create_collection(self, name):
        try:
            collection = self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
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
    def create_disease_to_hps_dict(collection_name: str = "hpoa"):
        """
        Creates a dictionary mapping diseases (OMIM IDs) to their associated HPO IDs.

        :param collection_name: Name of the collection in the database (default is "hpoa").
        :return: Dictionary with diseases as keys and lists of corresponding HPO IDs as values.
        """
        disease_to_hps_dict = {}
        results = collection_name.get(where={}, include=["metadatas"])
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

    # use this for metric
    @staticmethod
    def create_hpo_id_to_embedding(collection_name: str = "ont_hp") -> Dict[str, Dict]:
        """
        Create a dictionary mapping HPO IDs to embeddings.

        :param collection_name: The collection to process.
        :return: A dictionary mapping HPO IDs to a dictionary of their label and embeddings.
        """
        hpo_id_to_data = {}
        results = collection_name.get(where={}, include=["metadatas", "embeddings"])
        for metadata, embedding in zip(results.get("metadatas", []), results.get("embeddings", [])):
            metadata_json = json.loads(metadata['_json'])
            hpo_id = metadata_json.get("original_id")
            if hpo_id:
                hpo_id_to_data[hpo_id] = {"embeddings": embedding}  # #{'HP:0005872': [1,2,3, ...]}
        return hpo_id_to_data


    def upsert_from_ont_hp_into_hpoToEmbedding_collection(self) -> Collection:
        """
        Upserts embeddings for each HPO ID into the "HPOtoEmbeddings" collection.

        :return: The updated "HPOtoEmbeddings" collection.
        """
        cachedDict = self.create_hpo_id_to_embedding()
        hpoToEmbedding = self.get_collection("HPOtoEmbeddings")
        for hp, data in cachedDict.items():
            embedding_list = data['embeddings']
            hpoToEmbedding.upsert(ids=[hp], embeddings=[embedding_list], metadatas=[{"type": "HP"}])
        return hpoToEmbedding


    def upsert_from_ont_hp_into_hpembedding_collection(self) -> Collection:
        """
        Upserts embeddings for each HPO ID into the "HPOtoEmbeddings" collection.

        :return: The updated "HPOtoEmbeddings" collection.
        """
        cachedDict = self.create_hpo_id_to_embedding()
        hpoToEmbedding = self.get_collection("HpEmbeddings")
        for hp, data in cachedDict.items():
            embedding_list = data['embeddings']
            hpoToEmbedding.upsert(ids=[hp], embeddings=[embedding_list], metadatas=[{"type": "HP"}])
        return hpoToEmbedding

    def upsert_avgEmbeddings_into_diseaseAvgEmbeddings_collection(self) -> Collection:
        """
        Calculates and upserts the average embeddings for each disease into the "DiseaseAvgEmbeddings" collection.

        :param name: Name of the collection to be updated.
        :return: The updated "DiseaseAvgEmbeddings" collection.
        """
        ont_hp = self.get_collection("ont_hp")
        omimToHPdict = self.create_disease_to_hps_dict()
        diseaseAvgEmbedings = self.get_collection("DiseaseAvgEmbeddings")
        cachedDict = self.create_hpo_id_to_embedding(ont_hp)
        for disease, hps in omimToHPdict.items():
            average_embedding = self.calculate_average_embedding_from_cachedDict(hps, cachedDict)
            diseaseAvgEmbedings.upsert(ids=[disease], embeddings=[average_embedding.tolist()],
                                       metadatas=[{"type": "disease"}])
        return diseaseAvgEmbedings

    @staticmethod
    def calculate_average_embedding_from_cachedDict(hps, embeddings_dict):
        """
        Calculates the average embedding for a given set of HPO IDs.

        :param hps: List of HPO IDs.
        :param embeddings_dict: Dictionary mapping HPO IDs to their embeddings.
        :return: A numpy array representing the average embedding for the HPO IDs.
        """
        embeddings = [embeddings_dict[hp_id]['embeddings'] for hp_id in hps if hp_id in embeddings_dict]
        return np.mean(embeddings, axis=0) if embeddings else []

    def query_diseases_by_hpo_terms(self, hpo_ids: List[str]) -> List:
        """
        Queries the 'DiseaseAvgEmbeddings' collection for diseases closest to the average embeddings of given HPO terms.

        :param hpo_ids: List of HPO term IDs.
        :return: List of diseases sorted by closeness to the average HPO embeddings.
        """
        diseaseAvgEmbedings = self.get_collection("DiseaseAvgEmbeddings")
        ont_hp = self.get_collection("ont_hp")

        cachedDict = self.create_hpo_id_to_embedding(ont_hp)
        avg_embedding = self.calculate_average_embedding_from_cachedDict(hpo_ids, cachedDict)

        if avg_embedding is None:
            return "No valid embeddings found for provided HPO terms."

        query_results = diseaseAvgEmbedings.query(
            query_embeddings=[avg_embedding.tolist()],
            n_results=10,
            include=["embeddings", "distances"]
        )

        disease_ids = query_results['ids'][0] if 'ids' in query_results and query_results['ids'] else []
        distances = query_results['distances'][0] if 'distances' in query_results and query_results['distances'] else []
        sorted_results = sorted(zip(disease_ids, distances), key=lambda x: x[1])

        return sorted_results














        ####
    # use to compare the averaged vector from one disease and do cosine similarity with the vectors of the HPs that all belong to that disease
    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if magnitude == 0:
            return 0
        return dot_product / magnitude

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
    # deprecated cause dict structure
    def calculate_average_embedding(hps, embeddings_dict):
        embeddings = [embeddings_dict[hp_id] for hp_id in hps if hp_id in embeddings_dict]
        return np.mean(embeddings, axis=0) if embeddings else []


    # leave in case label should be included
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
