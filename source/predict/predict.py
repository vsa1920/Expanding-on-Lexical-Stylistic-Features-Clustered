'''Predict on a dataset.'''
import os
if "/shared/lyuqing/" in os.getcwd():
	os.environ["HF_HOME"] = "/shared/lyuqing/huggingface_cache"
if os.getcwd().endswith('predict'):
	os.chdir('../..')
import sys
sys.path.append('source')
from model.model import GoldModel, FreqFeaturizer, LexicalFeaturizer
from configuration.configuration import Config
from dataset.utils import load_dataset
from evaluate.evaluate import evaluate_acc
import csv
import argparse
from utils import PREDICTED_CLASS

if __name__ == "__main__":
	# config
	Parser = argparse.ArgumentParser()
	Parser.add_argument("--feature", help="the feature to extract", choices=["complexity", "formality", "intensity", "figurativeness"])
	Parser.add_argument("--dataset_name", help="the dataset name")
	Parser.add_argument("--split", help="the dataset split")
	Parser.add_argument("--model_name", help="model name")
	Parser.add_argument("--debug", help="debug mode. If enabled, only run on the first 50 examples.", action="store_true")

	# get the arguments
	args = Parser.parse_args()
	feature = args.feature
	dataset_name = args.dataset_name
	split = args.split
	model_name = args.model_name
	debug = args.debug

	config_frn = f"source/configuration/config_files/{feature}/{model_name}.json"
	config = Config.from_json_file(config_frn)

	# load dataset
	dataset_frn = f"data/{feature}/{dataset_name}/{split}.csv"
	train_frn = f"data/{feature}/{dataset_name}/train.csv"
	dataset = load_dataset(dataset_frn)
	dataset_train = load_dataset(train_frn)
	if debug:
		dataset = dataset[:50]
	
	if model_name == "gold":
		model = GoldModel()
	elif "freq" in model_name:
		model = FreqFeaturizer(config)
	else:
		model = LexicalFeaturizer(config)
		model.load_LM()
		model.generate_dvecs(dataset_train)

	# predict
	predicted_class = PREDICTED_CLASS[feature]

	preds = model.compare_feature_batch(dataset, feature)
	output_dir = f"output_dir/{feature}/{dataset_name}/{split}"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	output_fwn = f"{output_dir}/{model_name}{'_debug' if debug else ''}.csv"
	rows = []
	with open(output_fwn, 'w', encoding='utf-8') as fw:
		writer = csv.DictWriter(fw, fieldnames=list([f"gold_{predicted_class}", f"pred_{predicted_class}"]))
		writer.writeheader()
		for example, pred in zip(dataset, preds):
			row = {f"gold_{predicted_class}": example[f"gold_{predicted_class}"],
			       f"pred_{predicted_class}": pred}
			writer.writerow(row)
			rows.append(row)
	print(f"Predictions saved to {output_fwn}.")
	acc = evaluate_acc(rows, feature)
	print(f"Model: {model_name}\nAccuracy: {acc}")

