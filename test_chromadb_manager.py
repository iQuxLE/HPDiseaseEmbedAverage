from typing import Collection

import numpy as np
import pytest

from chromadb_manager import ChromaDBManager


@pytest.fixture
def db_manager() -> ChromaDBManager:
    db_manager = ChromaDBManager("/Users/carlo/Downloads/curate-gpt/db")
    return db_manager


@pytest.fixture
def ont_hp_collection(db_manager) -> Collection:
    return db_manager.client.get_collection("ont_hp")


@pytest.fixture
def cached_dict():
    return {
        'HP:0000118': {'embeddings': [1, 2, 3]},
        'HP:0000478': {'embeddings': [4, 5, 6]},
    }


def test_get_embeddings_by_hpo_ids_faster(db_manager, ont_hp_collection):
    hpo_id_to_data_dict = db_manager.create_hpo_id_to_data_dict_with_embedding(ont_hp_collection)
    mock_data = {
        'OMIM:619340': ['HP:0000118', 'HP:0000478', 'HP:0002643', 'HP:0001789', 'HP:0011097', 'HP:0000006',
                        'HP:0032792']
    }

    for hpo_id, hpo_data in mock_data.items():
        embeddings = db_manager.get_embeddings_by_hpo_ids_faster(hpo_data, hpo_id_to_data_dict)
        for hp in hpo_data:
            assert hp in embeddings, f"Embedding missing for HPO ID: {hp}"
            assert embeddings[hp] is not None, f"No embedding found for HPO ID: {hp}"
            print(f"Embeddings for HPO ID {hp}: {embeddings[hp]}")


def test_create_hpo_id_to_data_dict_with_embedding(db_manager, ont_hp_collection):
    hpo_id_to_data_dict = db_manager.create_hpo_id_to_data_dict_with_embedding(ont_hp_collection)

    assert hpo_id_to_data_dict is not None, "The returned dictionary is None"

    test_hpo_id = 'BSPO:0000102'
    expected_data = {
        'label': 'ventral to',
        'embeddings': [-0.02075735665857792, -0.006406246218830347, 0.018757153302431107, 0.009791205637156963]
    }

    assert test_hpo_id in hpo_id_to_data_dict, f"{test_hpo_id} is not in the dictionary"
    assert 'label' in hpo_id_to_data_dict[test_hpo_id], f"Label missing for {test_hpo_id}"
    assert 'embeddings' in hpo_id_to_data_dict[test_hpo_id], f"Embeddings missing for {test_hpo_id}"

    assert hpo_id_to_data_dict[test_hpo_id]['label'] == expected_data[
        'label'], f"Label for {test_hpo_id} does not match expected"

    actual_embeddings = hpo_id_to_data_dict[test_hpo_id]['embeddings'][:4]
    assert np.allclose(actual_embeddings,
                       expected_data['embeddings']), f"First four embeddings for {test_hpo_id} do not match expected"


def test_calculate_average_embedding_from_cachedDict(db_manager, cached_dict):
    mock_hps = ['HP:0000118', 'HP:0000478']
    expected_average = np.mean([[1, 2, 3], [4, 5, 6]], axis=0)

    actual_average = db_manager.calculate_average_embedding_from_cachedDict(mock_hps, cached_dict)

    assert np.allclose(actual_average, expected_average), "The calculated average embedding is not as expected"

