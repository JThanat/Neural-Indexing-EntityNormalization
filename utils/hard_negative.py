import os
import pandas as pd
import glob
import time

def get_id2name(dataset_path):
    all_name = pd.read_csv(dataset_path, delimiter='\t')
    all_name = all_name.dropna() 
    all_name = all_name[all_name['organism'] == 'HUMAN']
    
    id2name = {}
    for idx, row in all_name.iterrows():
        if row['id'].strip() not in id2name:
            id2name[row['id']] = set()
        id2name[row['id']].add(row['name'].strip())
        
    return id2name


def get_hard_neg(hard_neg_path):
    df_list = []
    for file in glob.glob(hard_neg_path + '/*'):
        df_list.append(pd.read_csv(file, delimiter='\t'))
    try:
        df = pd.concat(df_list)
        return df
    except Exception as e:
        df = pd.DataFrame(columns = ['name1', 'name2'])
        return df
      

def remove_dup_hard_neg(df, devset):
    name2neg = {}
    for i, row in df.iterrows():
        n1 = row['name1']
        n2 = row['name2']
        if n1 not in name2neg:
            name2neg[n1] = set()
        name2neg[n1].add(n2)
    
    hard_neg_train = []
    hard_neg_dev = []
    for k, v in name2neg.items():
        for ele in v:
            if k not in devset and ele not in devset:
                hard_neg_train.append([k, ele, 0.0])
            elif k in devset and ele in devset:
                hard_neg_dev.append([k, ele, 0.0])
    return hard_neg_train, hard_neg_dev


def get_name2id(dataset_path):
    all_name = pd.read_csv(dataset_path, delimiter='\t')
    all_name = all_name.dropna() 
    all_name = all_name[all_name['organism'] == 'HUMAN']
    
    name2id = {}
    for idx, row in all_name.iterrows():
        if row['name'].strip() not in name2id:
            name2id[row['name']] = set()
        name2id[row['name']].add(row['id'].strip())
        
    return name2id


def hard_neg_mining_from_ref(model, annoy_object, ref, id2name, top_k_hits=10):
    corpus_sentences = annoy_object['corpus_sentences']
    corpus_embeddings = annoy_object['corpus_embeddings']
    name_to_id = annoy_object['name_to_id']
    annoy_index = annoy_object['annoy_index']

    hard_negative_pairs = []
    for i, kv in enumerate(ref):
        query, answers = kv
        query_embedding = model.encode(query)

        found_corpus_ids, scores = annoy_index.get_nns_by_vector(query_embedding, top_k_hits, include_distances=True)
        hits = []
        for id, score in zip(found_corpus_ids, scores):
            hits.append({'corpus_id': id, 'score': 1-((score**2) / 2)})

        end_time = time.time()
        
        possible_answer = set()
        for ans in answers:
            possible_answer = possible_answer.union(id2name.get(ans, set()))
        
        for hit in hits[0:top_k_hits]:
            hitname = corpus_sentences[hit['corpus_id']]
            if hitname not in possible_answer:
                for name_in_corpus in possible_answer:
                    hard_negative_pairs.append([name_in_corpus, hitname])
            
    return hard_negative_pairs