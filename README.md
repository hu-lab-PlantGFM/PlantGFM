# PlantGFM

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3.10/)
[![PyTorch Version](https://img.shields.io/badge/torch-2.4.1-red.svg)](https://pytorch.org/get-started/locally/)
[![Transformers Version](https://img.shields.io/badge/transformers-4.44-orange.svg)](https://huggingface.co/transformers/)
[![Accelerate Version](https://img.shields.io/badge/accelerate-0.34-yellow.svg)](https://huggingface.co/docs/accelerate/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Welcome to the official repository for the paper "PlantGFM: A Genetic Foundation Model for Discovery and Creation of Plant Genes".

In this repository, you will find the following:

- Comprehensive guidelines for both pre-training and fine-tuning the models, including preprocessing steps and handling special cases in your data.
- Resources and example scripts to assist you in preparing your data and running the models for various tasks.
- The related code is in **`./plantgfm`**::
  - **`configuration_plantgfm.py`**: This is the configuration file for the PlantGFM model.
  - **`modeling_plantgfm.py`**: This file contains the model architecture for PlantGFM, along with code for classification and regression tasks.
  - **`modeling_segmentgfm.py`**: This script is focused on gene prediction tasks.

## 1. Environment 🚀

#### 1.1 Download and install [Anaconda](https://www.anaconda.com/download) package manager

#### 1.2 Create environment 

```bash
conda create -n gfms python=3.10
conda activate gfms
```

#### 1.3 Install dependencies

```bash
git clone --recursive https://github.com/hu-lab-PlantGFM/PlantGFM.git
cd PlantGFM
python3 -m pip install -r requirements.txt
```
## 2. Datasets 📊

To facilitate reproducibility, we provide **sample data** (demos) in this repository for format verification, and **full datasets** (or processing scripts) for reproduction.

| Scientific Task | Task Type | Sample Data <br> *(Format Reference)* | Full Data |
| :--- | :--- | :--- | :--- |
| **Pre-training** | CLM | [`./sample_data/Pre_Train`](./sample_data/Pre_Train) | (See Sec 2.1) |
| **Gene Prediction** | Segmentation | [`./sample_data/Gene_Pred...`](./sample_data/Gene_Prediction) | (See Sec 2.1) |
| **Gene Generation** | CLM | [`./sample_data/Gene_Gen...`](./sample_data/Gene_Generation) |[📥 **Download from Hugging Face**](https://huggingface.co/datasets/hu-lab/Gene_Generation) |
| **TFBS Prediction** | Classification | [`./sample_data/TFBS_...`](./sample_data/TFBS_Prediction) | [📥 **Download from Hugging Face**](https://huggingface.co/datasets/hu-lab/TFBS_Prediction) |
| **CRE Strength** | Regression | [`./sample_data/CREs_...`](./sample_data/CREs_Strength_Prediction) | [📥 **Download from Hugging Face**](https://huggingface.co/hu-lab/datasets/hu-lab/CREs_Strength_Prediction) |
| **Gene Expression** | Regression | [`./sample_data/Gene_Exp...`](./sample_data/Gene_Expression_Prediction) | [📥 **Download from Hugging Face**](https://huggingface.co/hu-lab/datasets/hu-lab/Gene_Expression_Prediction) |
| **Chromatin Access.** | Classification | [`./sample_data/Chromatin...`](./sample_data/Chromatin_Accessibility_Prediction_ZM) | [📥 **Download from Hugging Face**](https://huggingface.co/hu-lab/datasets/hu-lab/Chromatin_Accessibility_Prediction_ZM) |


> **Note**: The `sample_data` folder contains only mini-batches provided for debugging purposes.

#### 2.1 Data Processing (For Pre-training & Gene Prediction) 
Due to the large scale of genomic data, we provide the processing scripts to generate the training datasets from raw genomes (download links can be found in the Supplementary Materials of our manuscript).

* **Scripts Location**: Please refer to `./data_processing/`.

```bash
# Example 1: Processing raw genome for Pre-training (CLM)
# Splits genome into fixed-length sequences (65k bp) with overlap.
python data_processing/gene_prediction/segment_genome.py \
    --input raw_genome.fna \
    --output raw_genome.txt

# Example 2: Processing data for Gene Prediction (Segmentation)
# Note: Requires genome sequence (.fna) as the 1st argument and annotation file (.gff) as the 2nd.
# The script will create a folder containing a labeled '.tsv' file ready for fine-tuning.
bash data_processing/gene_prediction/run.sh \
    /path/to/raw_genome.fna \
    /path/to/raw_genome.gff
```  
## 3. Pre-train ✒️

If you wish to pre-train PlantGFM.To ensure compatibility with our pre-training scripts, your data needs to be formatted according to the structure in the `/sample/pre-data` directory.

```bash
python pre_train.py \
    --train_data_path './sample_data/Pre_Train/train.txt' \
    --dev_data_path './sample_data/Pre_Train/val.txt' \
    --tokenizer_path './tokenizer.json' \
    --max_length 65538 \
    --output_dir './output' \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --max_steps 30000 \
    --logging_steps 1250 \
    --save_steps 1250 \
    --eval_steps 1250 \
    --learning_rate 6e-4 \
    --gradient_accumulation_steps 24 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --warmup_steps 1000 \
    --lr_scheduler_type "cosine" \
    --save_total_limit 24 \
    --save_safetensors False \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --bf16 True \
    --seed 2024
```
### Pre-training Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| **Data & Model Paths** | | |
| `train_data_path` | `./sample_data/pre-train/train.txt` | Path to the training dataset file. |
| `dev_data_path` | `./sample_data/pre-train/dev.txt` | Path to the validation dataset file. |
| `tokenizer_path` | `/path/to/model` | Path to the pre-trained tokenizer. |
| `output_dir` | `./output` | Directory where model checkpoints will be saved. |
| `init_model_path` | `None` | (Optional) Path to initialize weights for staged pre-training. |
| **Training Hyperparameters** | | |
| `max_length` | `1024` | Maximum length of input sequences. |
| `learning_rate` | `6e-4` | Initial learning rate. |
| `per_device_train_batch_size` | `1` | Batch size per GPU for training. |
| `per_device_eval_batch_size` | `1` | Batch size per GPU for evaluation. |
| `gradient_accumulation_steps` | `24` | Number of updates steps to accumulate before backward pass. |
| `max_steps` | `30000` | Total number of training steps to perform. |
| `warmup_steps` | `1000` | Number of steps used for a linear warmup from 0 to learning_rate. |
| `weight_decay` | `0.1` | Weight decay to apply to all layers except bias/LayerNorm. |
| `adam_beta1` | `0.9` | Beta1 for AdamW optimizer. |
| `adam_beta2` | `0.95` | Beta2 for AdamW optimizer. |
| `lr_scheduler_type` | `"cosine"` | The scheduler type to use (`linear`, `cosine`, `constant`). |
| **System & Logging** | | |
| `bf16` | `True` | Whether to use bf16 precision (requires Ampere+ GPU). |
| `gradient_checkpointing` | `True` | Saves memory by recomputing gradients during backward pass. |
| `ddp_find_unused_parameters` | `False` | Set to True only if using DDP with unused parameters. |
| `save_safetensors` | `False` | Whether to save models in `safetensors` format. |
| `logging_steps` | `1250` | Number of update steps between logging training metrics (e.g., loss, learning rate). |
| `save_steps` | `1250` | Number of update steps between saving model checkpoints to the output directory. |
| `eval_steps` | `1250` | Number of update steps between performing evaluation on the validation set. |
| `save_total_limit` | `24` | Limit the total amount of checkpoints. Deletes the oldest. |
| `seed` | `2024` | Random seed for initialization to ensure reproducibility. |



## 4. Fine-tune ✏️

Whether you aim to **fine-tune** the model on custom datasets or **reproduce** the results reported in our manuscript, the first step is to download PlantGFM locally from Hugging Face🤗. 

Before proceeding, please note the following **critical configurations**: 🔍

- **Sequence Preprocessing**: The sequences need to be converted into individual nucleotides with spaces. For example, the sequence `"ATCGACCT"` must be processed into `"A T C G A C C T"`.
- **Handling Other Bases**: Although our model was pre-trained primarily on `'A', 'T', 'C', 'G', 'N'`, it allows for a small frequency of other characters.
- **Recommended Hyperparameters**: To ensure stable convergence and reproducibility, we recommend setting the learning rate to 1e-4 or 5e-5 and weight decay to 0.01, while keeping all other parameters at their default values.
- **Max Length**: Set `max_length` to `data_length + 2` (accounting for start and end tokens).
- **Training Dynamics**:
    - **Classification Tasks** (e.g., TFBS Prediction, Chromatin Accessibility): Converge rapidly, typically requiring only **a few epochs**.
    - **Regression Tasks** (e.g., Gene Expression, CRE Strength): Require an extended training horizon, usually needing **dozens of epochs** for full convergence.
- **Data Format**: Please ensure your data strictly follows the **CSV** (for Classification/Regression) or **TSV** (for Segmentation) formats provided in the **[Sample Data (Section 2)](#2-data-preparation--benchmarks-)**.

```bash
python fine_tune.py \
  --data_name './sample_data/CREs_Strength_Prediction/'  \
  --output_dir './output' \
  --model_name_or_path 'path/to/model' \
  --tokenizer_path './tokenizer.json' \
  --max_length 172 \
  --batch_size 32 \
  --epochs 10 \
  --learning_rate 1e-4 \
  --logging_strategy 'epoch' \
  --evaluation_strategy 'epoch' \
  --save_strategy 'epoch' \
  --save_total_limit 3 \
  --weight_decay 0.01 \
  --task_type 'regression' \
  --seed 2024
```


### Fine-tuning Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| **Data & Model Paths** | | |
| `data_name` | `None` | Path to the dataset folder containing train/dev/test files. |
| `output_dir` | `None` | Directory where checkpoints and logs will be saved. |
| `model_name_or_path` | `None` | Hugging Face model ID or local path to the pre-trained model. |
| `tokenizer_path` | `None` | Path to the tokenizer directory. |
| **Training Hyperparameters** | | |
| `task_type` | `None` | The specific task type: `classification`, `regression`, or `segmentation`. |
| `epochs` | `20` | Total number of training epochs. |
| `batch_size` | `96` | Batch size per device for training and evaluation. |
| `learning_rate` | `1e-4` | Initial learning rate. |
| `weight_decay` | `0.001` | Weight decay coefficient to prevent overfitting. |
| `max_length` | `172` | Max sequence length. Longer sequences are truncated. |
| **Strategy & Logging** | | |
| `logging_strategy` | `'epoch'` | Logging frequency (`steps` or `epoch`). |
| `evaluation_strategy` | `'epoch'` | Evaluation frequency (`steps` or `epoch`). |
| `save_strategy` | `'epoch'` | Checkpoint saving frequency (`steps` or `epoch`). |
| `save_total_limit` | `1` | Maximum number of checkpoints to keep (deletes oldest). |
| `seed` | `2024` | Random seed for initialization to ensure reproducibility. |






Feel free to contact us if you have any questions or suggestions regarding the code and models.
