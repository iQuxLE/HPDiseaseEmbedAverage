from typing import Collection

import numpy as np
import pytest

from core.chromadb_manager import ChromaDBManager


@pytest.fixture
def db_manager() -> ChromaDBManager:
    db_manager = ChromaDBManager("/Users/carlo/Downloads/curate-gpt/db")
    return db_manager


@pytest.fixture
def ont_hp_collection(db_manager) -> Collection:
    return db_manager.client.get_collection("ont_hp")


@pytest.fixture
def ont_hpoa_collection(db_manager) -> Collection:
    return db_manager.client.get_collection("hpoa")


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


def test_extraction_functions_produce_same_output(db_manager):
    file_based_dataset = "/Users/carlo/PycharmProjects/chroma_db_playground/test.hpoa"
    hpoa = db_manager.get_collection("hpoa")

    collection_based_dataset = hpoa.get(limit=1)

    file_based_output = db_manager.extract_and_use_omim_hpo_mappings(file_based_dataset)
    collection_based_output = db_manager.extract_omim_hpo_mappings_from_collection(collection_based_dataset)

    assert file_based_output == collection_based_output, "Outputs do not match"

def test_output_filebased(db_manager):
    file_based_dataset = "/Users/carlo/PycharmProjects/chroma_db_playground/test.hpoa"
    file_based_output = db_manager.extract_and_use_omim_hpo_mappings(file_based_dataset)
    print(file_based_output)


# def test_collection_based_output(db_manager, ont_hpoa_collection):
#     collection_based_dataset = ont_hpoa_collection.get(limit=1)
#     # print(collection_based_dataset)
#     collection_based_output = db_manager.extract_omim_hpo_mappings_from_collection_via_json(collection_based_dataset)
#     print(collection_based_output)
#
# def test_collection_based_output(db_manager, ont_hpoa_collection):
#     collection_based_dataset = ont_hpoa_collection.get(limit=1)
#     print(type(collection_based_dataset))
    # if collection_based_dataset:
    #     first_item = collection_based_dataset[0]  # Extract the first item from the list
    #     collection_based_output = db_manager.extract_omim_hpo_mappings_from_collection_via_json1(first_item)
    #     print(collection_based_output)

def test_collection_based_output(db_manager, ont_hpoa_collection):
    record = ont_hpoa_collection.get()
    collection_based_output = db_manager.extract_omim_hpo_mappings_from_dict_via_jsonstring(record)
    print(collection_based_output['OMIM:619340'])

def test_full_collection_output(db_manager, ont_hpoa_collection):
    full_collection_output = db_manager.extract_omim_hpo_mappings_from_entire_collection(ont_hpoa_collection)
    print(full_collection_output['OMIM:619340'])



# {'ids': ['{\'disease\': \'OMIM:619340\', \'disease_label\': \'Developmental and epileptic encephalopathy 96\', \'qualifier\': \'\', \'phenotype\': \'HP:0011097\', \'reference\': \'PMID:31675180\', \'evidence\': \'PCS\', \'onset\': \'\', \'frequency\': \'1/2\', \'sex\': \'\', \'modifier\': \'\', \'aspect\': \'P\', \'biocuration\': \'HPO:probinson[2021-06-21]\', \'phenotype_label\': \'Epileptic spasm\', \'publications\': [{\'id\': \'PMID:31675180\', \'title\': \'De novo NSF mutations cause early infantile epileptic encephalopathy.\', \'abstract\': \'N-ethylmaleimide-sensitive factor (NSF) plays a critical role in intracellular vesicle transport, which is essential for neurotransmitter release. Herein, we, for the first time, document human monogenic disease phenotype of de novo pathogenic variants in NSF, that is, epileptic encephalopathy of early infantile onset. When expressed in the developing eye of Drosophila, the mutant NSF severely affected eye development, while the wild-type allele had no detectable effect under the same conditions. Our findings suggest that the two pathogenic variants exert a dominant negative effect. De novo heterozygous mutations in the NSF gene cause early infantile epileptic encephalopathy.\', \'pmcid\': \'PMCID:PMC6856629\'}], \'_json\': \'{"disease": "OMIM:619340", "disease_label": "Developmental and epileptic encephalopathy 96", "qualifier": "", "phenotype": "HP:0011097", "reference": "PMID:31675180", "evidence": "PCS", "onset": "", "frequency": "1/2", "sex": "", "modifier": "", "aspect": "P", "biocuration": "HPO:probinson[2021-06-21]", "phenotype_label": "Epileptic spasm", "publications": [{"id": "PMID:31675180", "title": "De novo NSF mutations cause early infantile epileptic encephalopathy.", "abstract": "N-ethylmaleimide-sensitive factor (NSF) plays a critical role in intracellular vesicle transport, which is essential for neurotransmitter release. Herein, we, for the first time, document human monogenic disease phenotype of de novo pathogenic variants in NSF, that is, epileptic encephalopathy of early infantile onset. When expressed in the developing eye of Drosophila, the mutant NSF severely affected eye development, while the wild-type allele had no detectable effect under the same conditions. Our findings suggest that the two pathogenic variants exert a dominant negative effect. De novo heterozygous mutations in the NSF gene cause early infantile epileptic encephalopathy.", "pmcid": "PMCID:PMC6856629"}]}\'}'],
#  'embeddings': None,
#  'metadatas': [{'_json': '{"disease": "OMIM:619340", "disease_label": "Developmental and epileptic encephalopathy 96", "qualifier": "", "phenotype": "HP:0011097", "reference": "PMID:31675180", "evidence": "PCS", "onset": "", "frequency": "1/2", "sex": "", "modifier": "", "aspect": "P", "biocuration": "HPO:probinson[2021-06-21]", "phenotype_label": "Epileptic spasm", "publications": [{"id": "PMID:31675180", "title": "De novo NSF mutations cause early infantile epileptic encephalopathy.", "abstract": "N-ethylmaleimide-sensitive factor (NSF) plays a critical role in intracellular vesicle transport, which is essential for neurotransmitter release. Herein, we, for the first time, document human monogenic disease phenotype of de novo pathogenic variants in NSF, that is, epileptic encephalopathy of early infantile onset. When expressed in the developing eye of Drosophila, the mutant NSF severely affected eye development, while the wild-type allele had no detectable effect under the same conditions. Our findings suggest that the two pathogenic variants exert a dominant negative effect. De novo heterozygous mutations in the NSF gene cause early infantile epileptic encephalopathy.", "pmcid": "PMCID:PMC6856629"}]}',
#    'aspect': 'P',
#    'biocuration': 'HPO:probinson[2021-06-21]',
#    'disease': 'OMIM:619340',
#    'disease_label': 'Developmental and epileptic encephalopathy 96',
#    'evidence': 'PCS',
#    'frequency': '1/2',
#    'modifier': '',
#    'onset': '',
#    'phenotype': 'HP:0011097',
#    'phenotype_label': 'Epileptic spasm',
#    'qualifier': '',
#    'reference': 'PMID:31675180',
#    'sex': ''}],
#  'documents': ['disease: OMIM:619340\ndisease_label: Developmental and epileptic encephalopathy 96\nphenotype: HP:0011097\nreference: PMID:31675180\nevidence: PCS\nfrequency: 1/2\naspect: P\nbiocuration: HPO:probinson[2021-06-21]\nphenotype_label: Epileptic spasm\npublications:\n- id: PMID:31675180\n  title: De novo NSF mutations cause early infantile epileptic encephalopathy.\n  abstract: N-ethylmaleimide-sensitive factor (NSF) plays a critical role in intracellular\n    vesicle transport, which is essential for neurotransmitter release. Herein, we,\n    for the first time, document human monogenic disease phenotype of de novo pathogenic\n    variants in NSF, that is, epileptic encephalopathy of early infantile onset. When\n    expressed in the developing eye of Drosophila, the mutant NSF severely affected\n    eye development, while the wild-type allele had no detectable effect under the\n    same conditions. Our findings suggest that the two pathogenic variants exert a\n    dominant negative effect. De novo heterozygous mutations in the NSF gene cause\n    early infantile epileptic encephalopathy.\n  pmcid: PMCID:PMC6856629'],
#  'uris': None,
#  'data': None}



# #description: "HPO annotations for rare diseases [8181: OMIM; 47: DECIPHER; 4242 ORPHANET]"
# #version: 2023-10-09
# #tracker: https://github.com/obophenotype/human-phenotype-ontology/issues
# #hpo-version: http://purl.obolibrary.org/obo/hp/releases/2023-10-09/hp.json
# database_id	disease_name	qualifier	hpo_id	reference	evidence	onset	frequency	sex	modifier	aspect	biocuration
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0011097	PMID:31675180	PCS		1/2			P	HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0002187	PMID:31675180	PCS		1/1			P	HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0001518	PMID:31675180	PCS		1/2			P	HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0032792	PMID:31675180	PCS		1/2			P	HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0011451	PMID:31675180	PCS		1/2			P	HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0010851	PMID:31675180	PCS		2/2			P	HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0001789	PMID:31675180	PCS		1/2			P	HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0200134	PMID:31675180	PCS		2/2			P	HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0001522	PMID:31675180	PCS		1/2			C	HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0000006	PMID:31675180	PCS					I	HPO:probinson[2021-06-21];HPO:probinson[2021-06-21]
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0002643	PMID:31675180	PCS		2/2			P	HPO:probinson[2021-06-21]