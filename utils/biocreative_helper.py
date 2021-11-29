import os
import xml.etree.ElementTree as ET
import pandas as pd
import time
import glob

def parse_biocreative(biocpath):
    path_to_bioc = os.path.join(biocpath, 'BioIDtraining_2/caption_bioc/')
    file_list = sorted(os.listdir(path_to_bioc))
    file_list = [f for f in file_list if not f.startswith('._')]
    
    tree = ET.parse(path_to_bioc + file_list[0])
    root = tree.getroot()
    
    bioc_proteins_queries = {}
    query_answer_set = []
    for f in file_list:
        tree = ET.parse(path_to_bioc + f)
        root = tree.getroot()
        for e in root.findall("document"):
            for p in e.findall("passage"):
                for ann in p.findall("annotation"):
                    for infon in ann.findall("infon"):
                        if infon.get('key') == 'type' and infon.text.lower().startswith("uniprot"):
                            p = ann.find('text').text
                            uid_list = infon.text.split("|")
                            uid_list = [uid.replace('Uniprot:', '') for uid in uid_list]
                            query_answer_set.append((p, set(uid_list)))
                            if p not in bioc_proteins_queries:
                                bioc_proteins_queries[p] = set([])
                            bioc_proteins_queries[p] = bioc_proteins_queries[p].union(set(uid_list))
                            
    return bioc_proteins_queries, query_answer_set


def get_qa_set(train_name, query_answer_set):
    filtered_qa_set = []
    for q, ans in query_answer_set:
        if q in train_name:
            continue
        filtered_qa_set.append((q,ans))
    return filtered_qa_set


def get_qa_merged_answer_set(train_name, bioc_proteins_queries):
    filtered_qa_merged_set = {}
    for q, ans in bioc_proteins_queries.items():
        if q in train_name:
            continue
        filtered_qa_merged_set[q] = ans
    return filtered_qa_merged_set
        
def get_qa_non_merged_list(train_name, filtered_qa_merged_list):
    filtered_qa_non_merged_list = []
    for q, answer_list in filtered_qa_merged_list:
        if q in train_name:
            continue
        for ans in answer_list:
            filtered_qa_non_merged_list.append((q,set([ans])))
            
    return filtered_qa_non_merged_list


def evaluate_bioc_hit(model, annoy_object, top_k_hits, query_answer_set, check_ref_id=False, is_qa_dict=False):
    corpus_sentences = annoy_object['corpus_sentences']
    corpus_embeddings = annoy_object['corpus_embeddings']
    name_to_id = annoy_object['name_to_id']
    annoy_index = annoy_object['annoy_index']

    top1hit_list = set()
    top3hit_list = set()
    top5hit_list = set()
    top10hit_list = set()
    skipped_query = set()
    total = 0
    
    all_ref_id = []
    if check_ref_id:
        for ids in name_to_id.values():
            for id in ids:
                all_ref_id.append(id)
    all_ref_id = set(all_ref_id)
    
    if is_qa_dict:
        query_answer_set = query_answer_set.items()
    
    for query, gt_ids in query_answer_set:
        if check_ref_id and len(gt_ids.intersection(all_ref_id)) == 0:
            skipped_query.add(query)
            continue
        total += 1
        query_embedding = model.encode(query)
        found_corpus_ids, scores = annoy_index.get_nns_by_vector(query_embedding, top_k_hits, include_distances=True)
        hits = []
        for id, score in zip(found_corpus_ids, scores):
            # Cosine Distance is equivalent to Euclidean distance of normalized vectors = sqrt(2-2*cos(u, v))
            # cosine_dist = sqrt(2-2*cos(u,v))
            # Thus cos(u,v) = 1-(cosine_dist**2)/2
            hits.append({'corpus_id': id, 'score': 1-((score**2) / 2)})

        for hit in hits[0:1]:
            possible_id = set(name_to_id.get(corpus_sentences[hit['corpus_id']], []))
            if len(gt_ids.intersection(possible_id)) > 0:
                top1hit_list.add(query)

        for hit in hits[0:3]:
            possible_id = set(name_to_id.get(corpus_sentences[hit['corpus_id']], []))
            if len(gt_ids.intersection(possible_id)) > 0:
                top3hit_list.add(query)

        for hit in hits[0:5]:
            possible_id = set(name_to_id.get(corpus_sentences[hit['corpus_id']], []))
            if len(gt_ids.intersection(possible_id)) > 0:
                top5hit_list.add(query)

        for hit in hits[0:10]:
            possible_id = set(name_to_id.get(corpus_sentences[hit['corpus_id']], []))
            if len(gt_ids.intersection(possible_id)) > 0:
                top10hit_list.add(query)
                
    print("H@1: {}".format(len(top1hit_list)/total))
    print("H@3: {}".format(len(top3hit_list)/total))
    print("H@5: {}".format(len(top5hit_list)/total))
    print("H@10: {}".format(len(top10hit_list)/total))

    output = {}
    output['top1hit_list'] = top1hit_list
    output['top3hit_list'] = top3hit_list
    output['top5hit_list'] = top5hit_list
    output['top10hit_list'] = top10hit_list
    output['skipped_query'] = skipped_query

    return output


def find_highcosine_not_in_ann(query_embedding, corpus_embeddings, top_k_hits, found_corpus_ids):
    # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity
    # Here, we compute the recall of ANN compared to the exact results
    correct_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k_hits)[0]
    correct_hits_ids = set([hit['corpus_id'] for hit in correct_hits])

    #Compute recall
    ann_corpus_ids = set(found_corpus_ids)
    if len(ann_corpus_ids) != len(correct_hits_ids):
        print("Approximate Nearest Neighbor returned a different number of results than expected")

    recall = len(ann_corpus_ids.intersection(correct_hits_ids)) / len(correct_hits_ids)
    print("\nApproximate Nearest Neighbor Recall@{}: {:.2f}".format(top_k_hits, recall * 100))

    if recall < 1:
        print("Missing results:")
        for hit in correct_hits[0:top_k_hits]:
            if hit['corpus_id'] not in ann_corpus_ids:
                print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))
    print("\n\n========\n")

    
def print_hit(model, annoy_object, query_list, id2name, top_k_hits, num_print=None):
    corpus_sentences = annoy_object['corpus_sentences']
    corpus_embeddings = annoy_object['corpus_embeddings']
    name_to_id = annoy_object['name_to_id']
    annoy_index = annoy_object['annoy_index']
    if not num_print:
        num_print = len(query_list)
    for i, kv in enumerate(query_list):
        if i >= num_print:
            break
        start_time = time.time()

        query, answers = kv
        query_embedding = model.encode(query)

        found_corpus_ids, scores = annoy_index.get_nns_by_vector(query_embedding, top_k_hits, include_distances=True)
        hits = []
        for id, score in zip(found_corpus_ids, scores):
            # Cosine Distance is equivalent to Euclidean distance of normalized vectors = sqrt(2-2*cos(u, v))
            # cosine_dist = sqrt(2-2*cos(u,v))
            # Thus cos(u,v) = 1-(cosine_dist**2)/2
            hits.append({'corpus_id': id, 'score': 1-((score**2) / 2)})

        end_time = time.time()
        
        possible_answer = set()
        for ans in answers:
            possible_answer = possible_answer.union(id2name.get(ans, set()))
        print("="*50)
        print("Input question:", query)
        print("Input answer:", answers)
        print("Possible Ans in Corpus ({}): {}".format(answers, possible_answer))
        print("Results (after {:.3f} seconds):".format(end_time-start_time))
        for hit in hits[0:top_k_hits]:
            possible_id = name_to_id.get(corpus_sentences[hit['corpus_id']], [])
            print("\t{:.3f}\t{}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']], possible_id))
            
            
def evaluate_non_hit(model, annoy_object, qa_set, hit_object, id2name, num_print=100, compute_hard_neg=False, top_k_hits=10):
    corpus_sentences = annoy_object['corpus_sentences']
    corpus_embeddings = annoy_object['corpus_embeddings']
    name_to_id = annoy_object['name_to_id']
    annoy_index = annoy_object['annoy_index']
    all_ref_id = []
    for ids in name_to_id.values():
        for id in ids:
            all_ref_id.append(id)
    all_ref_id = set(all_ref_id)
    
    non_hit = []
    # for query, gt_ids in bioc_proteins_queries.items():
    for query, gt_ids in qa_set:
        if len(gt_ids.intersection(all_ref_id)) == 0:
            continue
        if query not in hit_object['top10hit_list'] and len(gt_ids.intersection(all_ref_id)) > 0:
            non_hit.append((query, gt_ids))
    non_hit_ref_id = []
    for i, ele in enumerate(non_hit):
        for id in ele[1]:
            non_hit_ref_id.append(id)
    non_hit_ref_id = set(non_hit_ref_id)
    
    print_hit(model, annoy_object, non_hit, id2name, top_k_hits, num_print)
    
    if compute_hard_neg:
        hard_neg_pairs = hard_neg_mining_from_list(model, annoy_object, non_hit, id2name)
        return hard_neg_pairs
    else:
        return []
    

def hard_neg_mining_from_list(model, annoy_object, query_list, id2name, top_k_hits=10):
    corpus_sentences = annoy_object['corpus_sentences']
    corpus_embeddings = annoy_object['corpus_embeddings']
    name_to_id = annoy_object['name_to_id']
    annoy_index = annoy_object['annoy_index']

    hard_negative_pairs = []
    for i, kv in enumerate(query_list):
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
            
        for name_in_corpus in possible_answer:
            for hit in hits[0:top_k_hits]:
                # append all wrong hit 
                hard_negative_pairs.append([name_in_corpus, corpus_sentences[hit['corpus_id']]])
                
    return hard_negative_pairs