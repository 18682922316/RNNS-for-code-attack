#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/26/22 9:51 AM
# @Author  : zhangjie z00534677
# @Site    : 
# @File    : eval.py.py
# @Software: PyCharm Community Edition

import torch
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

sim_dist = {"0.0-0.80":0, "0.80-0.85":0, "0.85-0.90":0, "0.80-0.95":0, "0.95-1.0":0}
diff_len_dist = {"0":0, "1":0, "2":0, "3":0, ">3":0}
diff_list = []
sim_list = []

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="", type=str,
                    help="")

parser.add_argument("--output", default="output.txt", type=str,
                    help="")
args = parser.parse_args()




def stat_sim(keys, values):
    key_mean_embedding_list = []
    value_mean_embedding_list = []
    for key, value in zip(keys, values):
        
        key_sub_words = tokenizer_mlm.tokenize(key)
        key_sub_words = [tokenizer_mlm.cls_token] + key_sub_words[:510] + [tokenizer_mlm.sep_token]
        key_input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(key_sub_words)])

        value_sub_words = tokenizer_mlm.tokenize(value)
        value_sub_words = [tokenizer_mlm.cls_token] + value_sub_words[:510] + [tokenizer_mlm.sep_token]
        value_input_ids_=torch.tensor([tokenizer_mlm.convert_tokens_to_ids(value_sub_words)])

        with torch.no_grad():
            key_embeddings = codebert_mlm.roberta(key_input_ids_.to(DEVICE))[0][0][1:-1]
            key_mean_embedding = torch.mean(key_embeddings, dim=0, keepdim=True).cpu().detach().numpy()[0]
            key_mean_embedding_list.append(key_mean_embedding)

            value_embeddings = codebert_mlm.roberta(value_input_ids_.to(DEVICE))[0][0][1:-1]
            value_mean_embedding = torch.mean(value_embeddings, dim=0, keepdim=True).cpu().detach().numpy()[0]
            value_mean_embedding_list.append(value_mean_embedding)

        sims = cosine_similarity(np.array([key_mean_embedding]), np.array([value_mean_embedding]))
        if sims[0]<=0.8:
            sim_dist["0.0-0.80"] = sim_dist.get("0.0-0.80",0) + 1
        elif  sims[0]<=0.85 and sims[0]>0.8:
            sim_dist["0.80-0.85"] = sim_dist.get("0.80-0.85", 0) + 1
        elif  sims[0]<=0.90 and sims[0]>0.85:
            sim_dist["0.85-0.90"] = sim_dist.get("0.85-0.90", 0) + 1
        elif  sims[0]<=0.95 and sims[0]>0.90:
            sim_dist["0.90-0.95"] = sim_dist.get("0.90-0.95", 0) + 1
        else:
            sim_dist["0.95-1.0"] = sim_dist.get("0.95-1.0", 0) + 1
        
        # print(key + "->" + value + str(sims[0]))
        sim_list.append(sims[0])

    return sim_dist, np.mean(sim_list)
        


def stat_diff_len(keys, values):
    for k, v  in zip(keys,values):
        # if len(v) >= len(k):
        diff_len = np.abs(len(v) - len(k))
        diff_list.append(diff_len)
        if diff_len==0:
            diff_len_dist["0"] = diff_len_dist.get("0", 0) + 1
        elif diff_len==1:
            diff_len_dist["1"] = diff_len_dist.get("1", 0) + 1
        elif diff_len==2:
            diff_len_dist["2"] = diff_len_dist.get("2", 0) + 1
        elif diff_len == 3:
            diff_len_dist["3"] = diff_len_dist.get("3", 0) + 1
        else:
            diff_len_dist[">3"] = diff_len_dist.get(">3", 0) + 1

    diff_mean = np.mean(diff_list)
    key_mean_len = np.mean([ len(key) for key in keys])
    value_mean_len = np.mean([ len(value) for value in values])
    return diff_len_dist, diff_mean, key_mean_len, value_mean_len


DEVICE = "cuda"
if __name__=="__main__":
    base_model = "/home/ma-user/work/attack_pre_code_model/CodeBERT/codebert-base/"
    codebert_mlm = RobertaForMaskedLM.from_pretrained(base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(base_model)
    codebert_mlm.to(DEVICE)
    
    

    fields = ['Index', 'Is Success', "Replaced Names"]
    input = "/home/ma-user/work/attack-pretrain-models-of-code-main/CodeXGLUE/Defect-detection-baseline/code/0_1000_attack_mhm.csv"
    index_input = pd.read_csv(input, skipinitialspace=True, usecols=fields)
    success = index_input[index_input['Is Success'] == 1]
    replaced_names_list = success['Replaced Names'].values.tolist()
    keys = []
    values = []
    for names in replaced_names_list:
        if names.startswith("{"):
            names = names[1:-1] 
        kvs = names.split(",")
        for kv in kvs:
            if ":" not in kv:
                continue

            k, v = kv.split(":")
            keys.append(k)
            values.append(v)


    diff_len_dist,  diff_mean,  key_mean_len, value_mean_len  = stat_diff_len(keys, values)
    sim_dist, sim_mean = stat_sim(keys, values)
    print(len(keys))
    print(key_mean_len)
    print(value_mean_len)
    print(diff_len_dist)
    print(diff_mean)
    print(sim_dist)
    print(sim_mean)





