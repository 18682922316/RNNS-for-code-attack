# Attack CodeBERT on Code Authorship Attribution Task

## Dataset

First, you need to download the dataset from [link](https://drive.google.com/file/d/1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe/view?usp=sharing). Then, you need to decompress the `.zip` file to the `dataset/`. For example:

```
pip install gdown
gdown https://drive.google.com/uc?id=1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe
unzip gcjpy.zip
mkdir dataset
cd dataset
mv ../gcjpy ./
```

Then, you can run the following command to preprocess the datasets:



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
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --valid_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
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
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --valid_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
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
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --valid_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee rnns_attacker.log
```


