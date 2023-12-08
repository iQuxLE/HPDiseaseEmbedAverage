from typing import Collection
import pytest

from chromadb_manager import ChromaDBManager


@pytest.fixture
def db_manager() -> ChromaDBManager:
    db_manager = ChromaDBManager("/Users/carlo/Downloads/curate-gpt/db")
    return db_manager


@pytest.fixture
def ont_hp_collection(db_manager) -> Collection:
    return db_manager.client.get_collection("ont_hp")


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
    # print(hpo_id_to_data_dict)
    assert hpo_id_to_data_dict is not None
    # test that 'BSPO:0000102': {'label': 'ventral to', 'embeddings': [-0.02075735665857792, -0.006406246218830347, 0.018757153302431107, 0.009791205637156963, ..}
    # is inside

# def test_get_embeddings_by_hpo_ids_faster():
#     fail()
#
#
# def test_get_record_by_hpo_id_faster():
#     fail()
