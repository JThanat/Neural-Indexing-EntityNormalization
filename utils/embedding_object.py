import abc
import os
import csv
import logging
import pickle

logger = logging.getLogger()


class EmbeddingObject:

    def __init__(self, corpus_sentences, corpus_embeddings, name_to_id):
        self.corpus_sentences = corpus_sentences
        self.corpus_embeddings = corpus_embeddings
        self.name_to_id = name_to_id


class EmbeddingObjectWrapper:

    def __init__(self, index_path, embedding_path, reference_dataset_path, name2id_path, model, embedding_size, max_corpus_size):
        self.index_path = index_path
        self.embedding_path = embedding_path
        self.reference_dataset_path = reference_dataset_path
        self.name2id_path = name2id_path
        self.model = model
        self.embedding_size = embedding_size
        self.max_corpus_size = max_corpus_size
        self.embedding_object = None

    @abc.abstractmethod
    def create_index(self):
        pass

    @abc.abstractmethod
    def create_embedding_and_index(self):
        pass

    def create_embedding(self, create_new = False):
        if not os.path.exists(self.embedding_path) or create_new:
            # Check if the dataset exists. If not, download and extract
            # Get all unique sentences from the file
            corpus_sentences = set()
            name_to_id = {}
            with open(self.reference_dataset_path, encoding='utf8') as infile:
                reader = csv.DictReader(infile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    name = row['name'].strip()
                    corpus_sentences.add(name)
                    if name not in name_to_id:
                        name_to_id[name] = []
                    name_to_id[name].append(row['id'].strip())

                    if len(corpus_sentences) >= self.max_corpus_size:
                        break

            corpus_sentences = list(corpus_sentences)
            logger.info("Encode the corpus of size {}. This might take a while".format(len(corpus_sentences)))
            corpus_embeddings = self.model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

            self.embedding_object = EmbeddingObject(corpus_sentences, corpus_embeddings, name_to_id)

            logger.info("Store file on disc")
            with open(self.embedding_path, "wb") as outfile:
                pickle.dump(self.embedding_object, outfile, pickle.HIGHEST_PROTOCOL)
        else:
            print("Load pre-computed embeddings from disc")
            with open(self.embedding_path, "rb") as infile:
                self.embedding_object = pickle.load(infile)

        return self.embedding_object

