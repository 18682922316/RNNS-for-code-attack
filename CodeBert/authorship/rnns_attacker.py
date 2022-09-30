
import sys
import os

import random
import copy
import json
import logging
import argparse
import warnings
import torch
import time
from tqdm import tqdm
import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel,T5Config, T5ForConditionalGeneration)


from model import Model
from run import TextDataset,InputFeatures
from utils import set_seed,select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, \
    get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue, set_seed, \
    get_masked_code_by_var, get_replaced_var_code_with_meaningless_char, CodeDataset
from python_parser.parser_folder import remove_comments_and_docstrings
from utils import Recorder
from python_parser.run_parser import get_identifiers, get_example, get_example_batch, remove_comments_and_docstrings



MODEL_CLASSES = {
    'CodeBert': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'GraphCodeBert': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'CodeT5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}


class RnnsAttacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.model_mlm = model_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score
        self.variable_emb, self.variable_name = self._get_variable_info()

    def _get_variable_info(self):
        codes = []
        variables = []
        variable_embs = []
        
        codebert_mlm = RobertaForMaskedLM.from_pretrained(self.args.base_model)
        tokenizer_mlm = RobertaTokenizer.from_pretrained(self.args.base_model)
        codebert_mlm.to('cuda')
        
        for path in [self.args.train_data_file, self.args.valid_data_file, self.args.test_data_file]:
            with open(path) as rf:
                for line in rf:
                    code = line.split(" <CODESPLIT> ")[0]
                    codes.append(code)

        for code in codes:
            try:
                identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(code, "python"), "python")
            except:
                identifiers, code_tokens = get_identifiers(code, "python")

            cur_variables = []
            for name in identifiers:
                if ' ' in name[0].strip() and not is_valid_variable_name(name[0], lang='python'):
                    continue
                cur_variables.append(name[0])

            variables.extend(cur_variables)

        variables = list(set(variables))
        for var in variables:
            sub_words = tokenizer_mlm.tokenize(var)
            sub_words = [tokenizer_mlm.cls_token] + sub_words[:self.args.block_size - 2] + [tokenizer_mlm.sep_token]
            input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
            with torch.no_grad():
                orig_embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0][0][1:-1]
                mean_embedding = torch.mean(orig_embeddings, dim=0, keepdim=True).cpu().detach().numpy()[0]
                variable_embs.append(mean_embedding)

        assert len(variable_embs) == len(variables)


        return variable_embs, variables
    
    def _convert_code_to_features(self, code, tokenizer, label, args):
        code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        return InputFeatures(source_tokens, source_ids, 0, label)
    
    def _get_var_importance_score_by_variance(self, args, example, code, words_list: list, sub_words: list, variable_names: list,
                                             tgt_model,
                                             tokenizer, batch_size=16, max_length=512,
                                             model_type='classification'):
        '''Compute the importance score of each variable'''
        # label: example[1] tensor(1)
        # 1. 过滤掉所有的keywords.
        positions = get_identifier_posistions_from_code(words_list, variable_names)
        # 需要注意大小写.
        if len(positions) == 0:
            ## 没有提取出可以mutate的position
            return None

        new_example = []

        # 2. 得到Masked_tokens
        masked_token_list, masked_var_list = get_replaced_var_code_with_meaningless_char(words_list, positions)
        # replace_token_positions 表示着，哪一个位置的token被替换了.

        for index, tokens in enumerate([words_list] + masked_token_list):
            new_code = ' '.join(tokens)
            new_feature = self._convert_code_to_features(new_code, tokenizer, example[1].item(), args)
            new_example.append(new_feature)
        new_dataset = CodeDataset(new_example)
        # 3. 将他们转化成features
        logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
        orig_probs = logits[0]
        orig_label = preds[0]

        var_pos_delt_prob_disp = {}
        var_neg_delt_prob_disp = {}
        var_importance_score_by_variance = {}
        for prob, var in zip(logits[1:], masked_var_list):
            if var in var_pos_delt_prob_disp:
                var_pos_delt_prob_disp[var].append(prob[orig_label])
                var_neg_delt_prob_disp[var].append(1 - prob[orig_label])
            else:
                var_pos_delt_prob_disp[var] = [prob[orig_label]]
                var_neg_delt_prob_disp[var] = [1 - prob[orig_label]]

        for var in var_pos_delt_prob_disp:
            MaxP = np.max(var_pos_delt_prob_disp[var] + var_neg_delt_prob_disp[var])
            VarP = (np.var(var_pos_delt_prob_disp[var]) + np.var(var_neg_delt_prob_disp[var])) / 2

            # var_importance_score_by_variance[var] = VarP/MaxP
            var_importance_score_by_variance[var] = VarP

        return var_importance_score_by_variance, positions


    def attack(self, example, code):
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[1].item()
        adv_code = ''
        temp_label = None
        true_label_prob = orig_prob[true_label]

        try:
            identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(code, "python"), "python")
        except:
            identifiers, code_tokens = get_identifiers(code, "python")

        variable_names = []
        for name in identifiers:
            if ' ' in name[0].strip() and not is_valid_variable_name(name[0], lang='python'):
                continue
            variable_names.append(name[0])
        
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        # 这里经过了小写处理..

        substituions = {}

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        sub_words = [self.tokenizer_tgt.cls_token] + sub_words[:self.args.block_size - 2] + [
            self.tokenizer_tgt.sep_token]

        names_to_importance_score, names_positions_dict = self._get_var_importance_score_by_variance(self.args, example,
                                                                                               processed_code,
                                                                                               words,
                                                                                               sub_words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.block_size,
                                                                                               model_type='classification')
        

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)

        ranking_order_variance = 0
        var_size = len(names_to_importance_score.keys())
        
        # 根据importance_score进行排序
        final_words = copy.deepcopy(words)
        final_code = copy.deepcopy(code)
        nb_changed_var = 0  # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        for name_and_score in sorted_list_of_names:
            used_candidate = list(replaced_words.values())
            tgt_word = name_and_score[0]
            tgt_word_len = len(tgt_word)
            tgt_positions = names_positions_dict[tgt_word]

            tgt_index = self.variable_name.index(tgt_word)
            distances = 1 - cosine_similarity(np.array(self.variable_emb),np.array([self.variable_emb[tgt_index]]))
            variable_index_list = [i for i,  distance in enumerate(distances) if distance < self.args.max_distance and len(self.variable_name[i])<= tgt_word_len + self.args.max_length_diff]
            variable_embs = [ self.variable_emb[index] for index in variable_index_list]
            valid_variable_names = [ self.variable_name[index] for index in variable_index_list]
            
            #random
            index_list = [i for i in range(0, len(valid_variable_names))]
            random.shuffle(index_list)
            inds_1 = index_list[:self.args.substitutes_size]
            inds_2 = index_list[self.args.substitutes_size:3*self.args.substitutes_size]
            all_substitues = [valid_variable_names[ind] for ind in inds_1]
            substituions[tgt_word] =  [valid_variable_names[ind] for ind in inds_2]

 
            # 得到了所有位置的substitue，并使用set来去重
            candidate = None
            new_substitutes = []
            for sub in all_substitues:
                if sub not in used_candidate:
                    new_substitutes.append(sub)
                       
            best_candidate = tgt_word
            loop_time = 0
            momentum = None
            track = []
            while True:
                # 依次记录了被加进来的substitue
                # 即，每个temp_replace对应的substitue.
                substitute_list = []
                replace_examples = []
                most_gap = 0.0
                for substitute in new_substitutes:
                    substitute_list.append(substitute)
                    temp_code = get_example(final_code, tgt_word, substitute, "python")
                    new_feature = self._convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
                    replace_examples.append(new_feature)
                if len(replace_examples) == 0:
                    # 并没有生成新的mutants，直接跳去下一个token
                    break

                new_dataset = CodeDataset(replace_examples)
                # 3. 将他们转化成features
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
                assert (len(logits) == len(substitute_list))
                used_candidate.extend(all_substitues)
                
                golden_prob_decrease_track = {}
                for index, temp_prob in enumerate(logits):
                    temp_label = preds[index]
                    if temp_label != orig_label:
                        # 如果label改变了，说明这个mutant攻击成功
                        is_success = 1
                        nb_changed_var += 1
                        nb_changed_pos += len(names_positions_dict[tgt_word])
                        candidate = substitute_list[index]
                        replaced_words[tgt_word] = candidate
                        adv_code = get_example(final_code, tgt_word, candidate, "python")
                        print("%s SUC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, candidate,
                               current_prob,
                               temp_prob[orig_label]), flush=True)
                        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                    else:
                        gap = current_prob - temp_prob[temp_label]
                        if gap > 0:
                            golden_prob_decrease_track[substitute_list[index]] = gap

                if len(golden_prob_decrease_track) > 0:
                    cur_iter_track = {}
                    sorted_golden_prob_decrease_track = sorted(golden_prob_decrease_track.items(), key=lambda x: x[1],
                                                               reverse=True)
                    (candidate, most_gap) = sorted_golden_prob_decrease_track[0]
                    cur_iter_track["candidate"] = list(map(lambda x: x[0], sorted_golden_prob_decrease_track[1:]))

                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    current_prob = current_prob - most_gap
                    if candidate not in valid_variable_names or best_candidate not in valid_variable_names:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, "python")
                        adv_code = final_code
                        break

                    candidate_index = valid_variable_names.index(candidate)
                    best_candidate_index = valid_variable_names.index(best_candidate)

                    if self.args.rnns_type == "RNNS-Raw":
                        prob_delt_emb = 0.0
                    else:
                        prob_delt_emb = variable_embs[candidate_index] - variable_embs[best_candidate_index]
                        
                    if momentum is None:
                        momentum = prob_delt_emb        
                    else:
                        momentum = (1 - self.args.a) * momentum + self.args.a * prob_delt_emb
                        
                    if self.args.rnns_type == "RNNS-Delta":
                        virtual_emb = variable_embs[candidate_index] + prob_delt_emb
                    else:
                        virtual_emb = variable_embs[candidate_index] + momentum
                        
                    similarity = cosine_similarity(np.array(variable_embs),
                                                   np.array([virtual_emb]))


                    inds = heapq.nlargest(1000, range(len(similarity)), similarity.__getitem__)
                    new_substitutes.clear()
                    if len(replaced_words) > 0:
                        used_candidate.extend(list(replaced_words.values()))

                    for ind in inds:
                        temp_var = valid_variable_names[ind]
                        if temp_var not in used_candidate:
                            new_substitutes.append(temp_var)
                            if temp_var in substituions[tgt_word]:
                                substituions[tgt_word].remove(temp_var)
                            used_candidate.append(temp_var)
                            if len(new_substitutes) >= self.args.substitutes_size:
                                break

                    best_candidate = candidate
                    cur_iter_track["best_candidate"] = best_candidate
                    track.append(cur_iter_track)

                else:
                    if best_candidate != tgt_word:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, "python")
                        adv_code = final_code
                        print("%s ACC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, best_candidate,
                               current_prob + most_gap,
                               current_prob), flush=True)

                    break

                loop_time += 1
                if loop_time >= self.args.iters:
                    if best_candidate != tgt_word:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, "python")
                        adv_code = final_code
                        print("%s ACC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, best_candidate,
                               current_prob + most_gap,
                               current_prob), flush=True)
                    break

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
    
    
    


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="train data file path.")
    parser.add_argument("--valid_data_file", default=None, type=str,
                        help="eval data file path.")  
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="test data file path.")
    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Path to store the CSV file")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    
    parser.add_argument("--number_labels", type=int,
                        help="The model checkpoint for weights initialization.")


    parser.add_argument("--tgt_model", default="/home/ma-user/work/attack_pre_code_model/CodeBERT/Authorship Attribution/model/model.bin", type=str,
                        help="")
    parser.add_argument("--max_distance", default=0.15, type=float)
    parser.add_argument("--max_length_diff", default=3, type=float)
    parser.add_argument("--substitutes_size", default=60, type=int) 
    parser.add_argument("--iters", default=6, type=int) 
    parser.add_argument("--a", default=0.2, type=float) 
    parser.add_argument("--rnns_type", default="RNNS-Smooth", type=str,  help="") 
    
    
    args = parser.parse_args()
    args.device = torch.device("cuda")

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    print("**************************************************************************")
    ## Load Target Model

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    print("**************************************************************************")
    config.num_labels=args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)
        
    model = Model(model,config,tokenizer,args)
    print("**************************************************************************")
    model.load_state_dict(torch.load(args.tgt_model))
    model.to(args.device)
    print("**************************************************************************")


    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to(args.device)

    ## Load Dataset
    test_dataset = TextDataset(tokenizer, args,args.test_data_file)

    # Load original source codes
    source_codes = []


    with open(args.test_data_file) as rf:
        for line in rf:
            code = line.split(" <CODESPLIT> ")[0]
            source_codes.append(code)
    assert (len(source_codes) == len(test_dataset))
    
    success_attack = 0
    total_cnt = 0
    recoder = Recorder(args.csv_store_path)
    
    attacker = RnnsAttacker(args, model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    start_time = time.time()
    query_times = 0
    masked_true_label_prob = 0.0
    sucessful_example_prob_cnt = []
    total_var_size = 0
    total_ranking_order_variance = 0
    for index, example in enumerate(test_dataset):
        example_start_time = time.time()
        code = source_codes[index]
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.attack(example, code)
        example_end_time = (time.time()-example_start_time)/60
        
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time()-start_time)/60, 2), "min")
        score_info = ''
        if names_to_importance_score is not None:
            for key in names_to_importance_score.keys():
                score_info += key + ':' + str(names_to_importance_score[key]) + ','

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','
        print("Query times in this attack: ", model.query - query_times)
        print("All Query times: ", model.query)
        recoder.write(index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success,
                      variable_names, score_info, nb_changed_var, nb_changed_pos, replace_info, args.rnns_type,
                      model.query - query_times, example_end_time)
        query_times = model.query
        
        if is_success >= -1 :
            # 如果原来正确
            total_cnt += 1

        if is_success == 1:
            success_attack += 1

        if total_cnt == 0:
            continue
        print("Success rate: ", 1.0 * success_attack / total_cnt)
        print("Successful items count: ", success_attack)
        print("Total count: ", total_cnt)
        print("Index: ", index)



if __name__ == '__main__':
    main()



