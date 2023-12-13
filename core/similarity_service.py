from typing import List
import numpy as np


class CosineSimilarityService:
    @staticmethod
    def calculate_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
        """ Calculate the cosine similarity between two vectors. """
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        return dot_product / (norm_a * norm_b)
