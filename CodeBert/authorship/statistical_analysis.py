
import torch
import pandas as pd
import numpy as np
import argparse
import json 
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,  AutoTokenizer, AutoModel)


parser = argparse.ArgumentParser()
parser.add_argument("--csv_store_path", default="", type=str,
                    help="")

args = parser.parse_args()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



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
    diff_variance = np.var(diff_list)
    key_mean_len = np.mean([ len(key) for key in keys])
    value_mean_len = np.mean([ len(value) for value in values])
    return diff_len_dist, diff_mean, key_mean_len, value_mean_len, diff_variance


DEVICE = "cuda"
if __name__=="__main__":
    
    fields = ['Index', 'Is Success', "Replaced Names"]
    inputs = ["0_1000_attack_mhm_adv.csv","rnns_attacker.csv"] 
    

    diff_len_dist = {"0":0, "1":0, "2":0, "3":0, ">3":0}
    diff_list = []
    sim_list = []
    var_list = []

    index_input = pd.read_csv(args.csv_store_path, skipinitialspace=True, usecols=fields)
    success = index_input[index_input['Is Success'] == 1]
    replaced_names_list = success['Replaced Names'].values.tolist()
    keys = []
    values = []

    for names in replaced_names_list:
        if names.startswith("{"):
            names = names[1:-1]
            kvs = names.split(",")
            var_list.append(len(kvs))
            for kv in kvs:
                if ":" not in kv:
                    continue
                k, v = kv.split(":")
                keys.append(k.strip()[1:-1])
                values.append(v.strip()[1:-1])
        else:
            kvs = names.split(",")
            var_list.append(len(kvs))
            for kv in kvs:
                if ":" not in kv:
                    continue
                k, v = kv.split(":")
                keys.append(k.strip())
                values.append(v.strip())

    var_num_mean = np.mean(var_list)
    var_num_var = np.var(var_list)
    diff_len_dist,  diff_mean,  key_mean_len, value_mean_len,diff_variance  = stat_diff_len(keys, values)


    print("**********************************************************************************")

    print("original variable length mean: " + str(key_mean_len))
    print("replaced variable length mean: "  + str(value_mean_len))
    print("variable length difference mean: "+ str(diff_mean))
    print("variable length difference variance: "+ str(diff_variance))

    print("replaced variable number mean: "+ str(var_num_mean))
    print("replaced variable number variance: "+ str(var_num_var))
    print("**********************************************************************************")



