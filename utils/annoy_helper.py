from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import torch
import logging
from annoy import AnnoyIndex
from utils.embedding_object import EmbeddingObjectWrapper

logger = logging.getLogger()


class AnnoyObjectWrapper(EmbeddingObjectWrapper):

    def __init__(self, index_path, embedding_path, reference_dataset_path, name2id_path, model, n_trees, embedding_size, max_corpus_size):
        super().__init__(index_path, embedding_path, reference_dataset_path, name2id_path, model, embedding_size, max_corpus_size)
        self.n_trees = n_trees
        self.annoy_index = None

    def create_index(self, create_new=False):
        if not os.path.exists(self.index_path) or create_new:
            # Create Annoy Index
            logger.info("Create Annoy index with {} trees. This can take some time.".format(self.n_trees))
            self.annoy_index = AnnoyIndex(self.embedding_size, 'angular')

            for i in range(len(self.embedding_object.corpus_embeddings)):
                self.annoy_index.add_item(i, self.embedding_object.corpus_embeddings[i])

            self.annoy_index.build(self.n_trees)
            self.annoy_index.save(self.index_path)
        else:
            # Load Annoy Index from disc
            self.annoy_index = AnnoyIndex(self.embedding_size, 'angular')
            self.annoy_index.load(self.index_path)

        return self.annoy_index

    def create_embedding_and_index(self, create_new_embedding=False, create_new_index=False):
        self.create_embedding(create_new=create_new_embedding)
        self.create_index(create_new=create_new_index)


def evaluate_embedding(annoy_object_wrapper, to_eval_data, top_k_hits=10, show_n_results=20):
    corpus_embeddings = torch.from_numpy(annoy_object_wrapper.embedding_object.corpus_embeddings)
    annoy_index = annoy_object_wrapper.annoy_index
    name_to_id = annoy_object_wrapper.name_to_id
    corpus_sentences = annoy_object_wrapper.embedding_object.corpus_sentences

    count = 0
    for i, triplet in enumerate(to_eval_data):
        if count >= show_n_results:
            break
        start_time = time.time()

        query, answer, label = triplet
        if label == 0:
            # skip negative pair as we won't evaluate it for now
            continue

        count += 1
        query_embedding = annoy_object_wrapper.model.encode(query)

        found_corpus_ids, scores = annoy_index.get_nns_by_vector(query_embedding,
                                                                 top_k_hits,
                                                                 include_distances=True)
        hits = []
        for _id, score in zip(found_corpus_ids, scores):
            # Cosine Distance is equivalent to Euclidean distance of normalized vectors = sqrt(2-2*cos(u, v))
            # cosine_dist = sqrt(2-2*cos(u,v))
            # Thus cos(u,v) = 1-(cosine_dist**2)/2
            hits.append({'corpus_id': _id, 'score': 1 - ((score ** 2) / 2)})

        end_time = time.time()

        logger.info("Input question:", query)
        logger.info("Results (after {:.3f} seconds):".format(end_time - start_time))
        for hit in hits[0:top_k_hits]:
            possible_id = name_to_id.get(corpus_sentences[hit['corpus_id']], [])
            logger.info("\t{:.3f}\t{}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']], possible_id))

        # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity
        # Here, we compute the recall of ANN compared to the exact results
        correct_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k_hits)[0]
        correct_hits_ids = set([hit['corpus_id'] for hit in correct_hits])

        # Compute recall
        ann_corpus_ids = set(found_corpus_ids)
        if len(ann_corpus_ids) != len(correct_hits_ids):
            logger.info("Approximate Nearest Neighbor returned a different number of results than expected")

        recall = len(ann_corpus_ids.intersection(correct_hits_ids)) / len(correct_hits_ids)
        logger.info("\nApproximate Nearest Neighbor Recall@{}: {:.2f}".format(top_k_hits, recall * 100))

        if recall < 1:
            logger.info("Missing results:")
            for hit in correct_hits[0:top_k_hits]:
                if hit['corpus_id'] not in ann_corpus_ids:
                    logger.info("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))
        logger.info("\n\n========\n")
