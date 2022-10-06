# Attack CodeBERT on Code Authorship Attribution Task

## Dataset

First, you need to download the train.txt and valid.txt from [link](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main/CodeXGLUE/Authorship-Attribution/dataset/data_folder/processed_gcjpy/)
Then, you need create the dataset directory and move train.txt,valid.txt to this directory.

## Code Parser
First, you need to download the code parser from [link](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main/python_parser)
Then, you need move python_parser directory to current directory.
Last, you python_parser/parser_folder


## Build `tree-sitter`

We use `tree-sitter` to parse code snippets and extract variable names. 
First, you need to download the code parser from [link](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main/python_parser)
Then, you need move python_parser directory to current directory.
Last, You need to go to `./python_parser/parser_folder` folder and build tree-sitter using the following commands: 
```
bash build.sh
```

## Attack
### On Python dataset

you can download the victim model into `saved_models/checkpoint-best-f1` by [this link](https://drive.google.com/file/d/14dOsW-_C0D1IINP2J4l2VqB-IAlGB15w/view?usp=sharing).

```shell
pip install gdown
mkdir saved_models/checkpoint-best-f1
gdown https://drive.google.com/uc?id=14dOsW-_C0D1IINP2J4l2VqB-IAlGB15w
mv model.bin saved_models/checkpoint-best-f1/
```


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


