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
## 2. Datasets & Benchmarks 📊

To ensure reproducibility while maintaining a lightweight repository, we distinguish between **Sample Data** and **Full Benchmarks**.

### 2.1 Sample Data (For Format Checking) 📂
The `./sample_data/` folder in this repository contains **mini-batch examples (demos)**. 
* **Purpose**: Use these files to understand the input file structure (headers, separators) and to test if the code runs successfully on your machine.
* **Warning**: Do **not** use these samples for training final models, as they are insufficient for convergence.

### 2.2 Full Datasets (For Reproduction) 🤗
The complete datasets used to produce the results in the paper are hosted on Hugging Face.
* **Location**: [**Hugging Face: hu-lab/datasets**](https://huggingface.co/hu-lab/datasets)
* **Access**: Please download the full datasets for the specific task you wish to reproduce.

### 2.3 Task-Data Mapping
Use the table below to find the correct data for your task:

| Scientific Task | Task Type | Sample Data (GitHub) <br> *(Format Reference)* | Full Dataset (Hugging Face) <br> *(Download for Training)* |
| :--- | :--- | :--- | :--- |
| **TFBS Prediction** | Classification | [`./sample_data/TFBS_Prediction`](./sample_data/TFBS_Prediction) | [Download Link](https://huggingface.co/hu-lab/datasets) |
| **CRE Strength** | Regression | [`./sample_data/CREs_Strength_Prediction`](./sample_data/CREs_Strength_Prediction) | [Download Link](https://huggingface.co/hu-lab/datasets) |
| **Gene Prediction** | Segmentation | [`./sample_data/Gene_Prediction`](./sample_data/Gene_Prediction) | [Download Link](https://huggingface.co/hu-lab/datasets) |
| **Gene Expression** | Regression | [`./sample_data/Gene_Expression_Prediction`](./sample_data/Gene_Expression_Prediction) | [Download Link](https://huggingface.co/hu-lab/datasets) |
| **Chromatin Access.** | Regression | [`./sample_data/Chromatin_Accessibility_Prediction_ZM`](./sample_data/Chromatin_Accessibility_Prediction_ZM) | [Download Link](https://huggingface.co/hu-lab/datasets) |

### 2.4 How to Use
1. **Download** the full dataset from Hugging Face.
2. **Organize** the data locally. We recommend following the same directory structure as `sample_data`.
3. **Update Arguments**: When running `fine_tune.py`, ensure the `--data_name` argument points to your downloaded full dataset, not the sample folder.
   ```bash
   # Example
   python fine_tune.py --data_name /path/to/downloaded/TFBS_Full_Data ...

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
    --bf16 True



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



## 4. Fine-tune ✏️
If you wish to fine-tune our model, the first step is to download PlantGFM locally from Hugging Face🤗. Before proceeding, please note the following important points:🔍


-**`Sequence Preprocessing`**: The sequences need to be converted into individual nucleotides. For example, the sequence "ATCGACCT" should be processed into "A T C G A C C T". between single nucleotides.

-**`Handling  Other Bases`** :  Although our model was pre-trained on the bases 'A', 'T', 'C', 'G', and 'N', it can also handle a small amount of other characters.

####  Classification and Regression

For both classification and regression tasks,your dataset should be formatted as a CSV file with the following structure:
 ```csv
sequence,labels
```

#### Segmentation

For segmentation tasks, your dataset should be formatted as a TSV file with the following structure:
 ```tsv
sequence    gene_0    gene_1    ...    gene_65536
```
Ensure that your data follows this structure, similar to the examples provided in `/sample_data/segmentation`, before proceeding with fine-tuning the model using the provided scripts.

Ensure that your data follows this structure, similar to the examples provided in `/sample_data/classification` and `/sample_data/regression`, before proceeding with fine-tuning the model using the provided scripts.

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
  --task_type 'regression'


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







Feel free to contact us if you have any questions or suggestions regarding the code and models.
