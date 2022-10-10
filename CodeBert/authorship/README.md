# Attack CodeBERT on Code Authorship Attribution Task

## Dataset

First, you need to download the train.txt and valid.txt from [link](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main/CodeXGLUE/Authorship-Attribution/dataset/data_folder/processed_gcjpy/).(This link is our reference work and will not reveal our identity).

Then, you need create the dataset directory and move train.txt,valid.txt to this directory.



## Build `tree-sitter`

We use `tree-sitter` to parse code snippets and extract variable names. 

First, you need to download the code parser from [link](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main/python_parser) .(This link is our reference work and will not reveal our identity).

Then, you need move python_parser directory to current directory.

Last, You need to go to `./python_parser/parser_folder` folder and build tree-sitter using the following commands: 

```
bash build.sh
```

## Fine-tune CodeBERT to get the victim model  
We use full train data for fine-tuning CodeBERT and valid data for evaluating.

```shell
python run.py \
    --output_dir=./saved_models/ \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --number_labels 66 \
    --do_train \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/valid.txt \
    --epoch 30 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
```
## Attack

#### RNN-Smooth attack

```shell
python rnns_attacker.py \
    --output_dir=./saved_models/ \
    --model_type=CodeBert \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --base_model=microsoft/codebert-base-mlm \
    --tgt_model=./saved_models/checkpoint-best-f1/model.bin \
    --rnns_type=RNNS-Smooth \
    --max_distance=0.15 \
    --max_length_diff=3 \
    --substitutes_size=60  \
    --iters=6 \
    --a=0.2 \
    --number_labels 66 \
    --csv_store_path ./rnns_attacker.csv \
    --language_type python \
    --train_data_file=../dataset/train.txt \
    --valid_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/valid.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee rnns_attacker.log
```

#### RNNS-Raw attack

```shell
python rnns_attacker.py \
    --output_dir=./saved_models/ \
    --model_type=CodeBert \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --base_model=microsoft/codebert-base-mlm \
    --tgt_model=./saved_models/checkpoint-best-f1/model.bin \
    --rnns_type=RNNS-Raw \
    --max_distance=0.15 \
    --max_length_diff=3 \
    --substitutes_size=60  \
    --iters=6 \
    --a=0.2 \
    --number_labels 66 \
    --csv_store_path ./rnns_attacker.csv \
    --language_type python \
    --train_data_file=../dataset/train.txt \
    --valid_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/valid.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee rnns_attacker.log
```

#### RNNS-Delta attack
```shell
python rnns_attacker.py \
    --output_dir=./saved_models/ \
    --model_type=CodeBert \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --base_model=microsoft/codebert-base-mlm \
    --tgt_model=./saved_models/checkpoint-best-f1/model.bin \
    --rnns_type=RNNS-Delta \
    --max_distance=0.15 \
    --max_length_diff=3 \
    --substitutes_size=60  \
    --iters=6 \
    --a=0.2 \
    --number_labels 66 \
    --csv_store_path ./rnns_attacker.csv \
    --language_type python \
    --train_data_file=../dataset/train.txt \
    --valid_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/valid.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee rnns_attacker.log
```


