# Lexical-Stylistic-Features
Code and data accompanying our paper "Representation of Lexical Stylistic Features in Language Models' Embedding Space" to appear at [*SEM 2023](https://sites.google.com/view/starsem2023).

ArXiv link coming soon!

## Get started

We suggest using miniconda/conda to set up the environment. The `environment.yml` file specifies the minimal dependencies.
You can create a virtual environment using it according to [this guildeline](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Essentially, you'll need to do something like:

```
cd /path/to/Lexical-Stylistic-Features
conda env create -f ./environment.yml --prefix ./envs
```

Afterwards, you'll also need to manually install the `en-core-web-md==3.5.0` English pipeline for spaCy using [this command](https://spacy.io/models/en#en_core_web_md), since it doesn't allow automatic installation with conda.

## Repository Structure
Here's a brief description of the repository structure and the function of each file.

- `environment.yml`: The conda environment file.
- `data/`: The data files. 
  - `complexity/`: The data for complexity extraction.
    - `dvecs/`: The vectors representing complexity, generated from each configuration (see `configuration/` below): e.g. `bert-base-multilingual-uncased_-1_seeds0_dvec.pkl` (the dves generated from `bert-base-multilingual-uncased`, last layer, using seeds pair set #0.).
    - `seeds/`: The seed pairs used to generate the dvecs.
    - `SimplePPDB`: The SimplePPDB dataset for evaluation, containing pairs of **words/phrases**.
      - `test.csv`: The processed test set.
      - `val.csv`: The processed validation set.
      - `raw/`: The raw data before processing. You don't need to worry about this.
    - `SimpleWikipedia`: The SimpleWikipedia dataset for evaluation, containing pairs of **sentences**.
      - `test.csv`: The test set.
      - `val.csv`: The validation set.
      - `raw/`: The raw data before processing. You don't need to worry about this.
  - `formality/`: The data for formality extraction (the file structure is the same as `complexity/`).
    - `GYAFC`: This dataset is proprietary and cannot be publicly shared. You can request access to it [here](https://github.com/raosudha89/GYAFC-corpus).
  - `intensity/`: The data for intensity extraction (the file structure is the same as `complexity/`).
  - `figurative/`: The data for figurative extraction (the file structure is the same as `complexity/`).
  - `concreteness/`: The data for concreteness extraction (in development).
- `output_dir/`: The directory to save model predictions. Each feature has one subdirectory.
- `source/`: Source code.
  - `utils.py`: Utility functions for the entire project.
  - `dataset/`: Dataset preprocessing / splitting scripts.
    - `extract_data.py`: Extract pairs of texts from the raw data files. When you add a new dataset, you should add into this file. See detailed comments in the file.
    - `split_data.py`: Split the extracted pairs into (train/)dev/test sets. When you add a new dataset, you need to minimally modify a few arguments in this file. See detailed comments in the file. 
    - `utils.py`: Utility functions for dataset processing.
  - `configuration/`: model configuration.
    - `configuration.py`: the Config class. See the definition of each field in the init() funciton.
    - `config_files/`: the configuration files for each model. Each feature has a subdirectory (e.g. `complexity/`), and there is also another subdirectory for combined features (`combined/`). The file name is in the format of `{model_name}.json`, e.g., `bert-base-multilingual-uncased_0_layeragg_seeds7_max.json`.
  - `model/`: The model scripts.
    - `model.py`: The model classes: GoldModel (the gold baseline), FreqFeaturizer (the frequency baseline), and LexicalFeaturizer (our model). See detailed comments inside.
  - `predict/`: The prediction scripts. See detailed instructions in the `Usage` section below.
    - `predict.py`: Run the model to make predictions on a dataset. Output will be saved under `ouput_dir/`. 
    - `logs/`: Log files for running the bash scripts.
  - `evaluate/`:
    - `evaluate.py`: Evaluate the model predictions with accuracy score.


## Usage

Before running a script, you should modify the `HF_HOME` variable in it (if present) to the path of your HuggingFace cache directory. This path should have large enough storage (for a lot of Huggingface model checkpoints). 

### Add a new feature / dataset
1. Make relevant folders under `dataset/` and put the new raw dataset there.
2. Modify the `extract_data.py` and `split_data.py` files to extract the clean pairs from the raw dataset and split them.
3. Make sure that the resulting files are named `test.csv` and `val.csv`, and are in the same format as the other datasets. See `data/complexity/SimplePPDB/test.csv` for an example.
4. If needed, write seed pairs for the new feature and put them under `data/{feature}/seeds/`. See `data/complexity/seeds/seeds_7.csv` for an example.

### Add a new configuration / Modify the model
1. To add a new configuration (e.g. hyperparameter combination), make a new file under `source/configuration/config_files/` with a new name `{model_name}.json`. See `complexity/bert-base-multilingual-uncased_0_layeragg_seeds7_max.json` for an example. You can define some new fields as needed (if so, you should modify the `configuration.py` file as needed to add them to the Config class.)
2. Modify `model/model.py` accordingly.

### Make predictions & Evaluate the model
Run `source/predict/predict.py` with the specified arguments (see the file for details). The output will be saved under `output_dir/` and the accuracy will also be printed to stdout.

Example: 
```
cd source/predict
mkdir -p logs/complexity/SimplePPDB/val
nohup python predict.py --feature "complexity" --dataset_name "SimplePPDB" --split "val" --model_name "bert-base-multilingual-uncased_0_layeragg_seeds7_max" > logs/complexity/SimplePPDB/val/bert-base-multilingual-uncased_0_layeragg_seeds7_max.log 2>&1 &
```

Output:
```
Predictions saved to output_dir/complexity/SimplePPDB/val/bert-base-multilingual-uncased_0_layeragg_seeds7_max.csv.
Model: bert-base-multilingual-uncased_0_layeragg_seeds7_max
Accuracy: 0.837
```
It's recommended to use nohup since some experiments take long.

### Evaluate existing predictions
If you already run `predict.py` but forget the accuracy, you can run `source/evaluate/evaluate.py` with the specified arguments (see the file for details) to get it again. The result will be printed to stdout.

Example:
```
python evaluate.py --feature "complexity" --dataset_name "SimplePPDB" --split "val" --model_name "bert-base-multilingual-uncased_0_layeragg_seeds7_max"
```
Output:
```
Model: bert-base-multilingual-uncased_0_layeragg_seeds7_max
Accuracy: 0.837
```

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{lyu-2023-representation,
    title = "Representation of Lexical Stylistic Features in Language Models' Embedding Space",
    author = "Lyu, Qing and Apidianaki, Marianna and Callison-Burch, Chris",
    booktitle = "Proceedings of the 112h Joint Conference on Lexical and Computational Semantics",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics"
}
```
