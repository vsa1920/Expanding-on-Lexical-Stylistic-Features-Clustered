'''Evaluate the model predictions.'''
import os
if os.getcwd().endswith('evaluate'):
	os.chdir('../..')
import sys
sys.path.append('source')
from dataset.utils import load_dataset, PREDICTED_CLASS_MAP
import argparse

def evaluate_acc(predictions, feature):
	'''Evaluate the accuracy of the predictions.'''
	total, correct = 0, 0
	predicted_class = PREDICTED_CLASS_MAP[feature]

	for i, row in enumerate(predictions):
		gold = int(row[f"gold_{predicted_class}"])
		pred = int(row[f"pred_{predicted_class}"])
		if gold == pred:
			correct += 1
		total += 1
	acc = round(correct / total, 3)
	return acc

if __name__ == "__main__":
	# config
	Parser = argparse.ArgumentParser()
	Parser.add_argument("--feature", help="the feature to extract", choices=["complexity", "formality", "intensity", "figurativeness"])
	Parser.add_argument("--dataset_name", help="the dataset name")
	Parser.add_argument("--split", help="the configuration file", choices=["seeds", "val", "test"])
	Parser.add_argument("--model_name", help="model name")
	Parser.add_argument("--debug", help="debug mode", action="store_true")

	# get the arguments
	args = Parser.parse_args()
	feature = args.feature
	split = args.split
	model_name = args.model_name
	dataset_name = args.dataset_name
	debug = args.debug

	# load predictions
	pred_frn = f"output_dir/{feature}/{dataset_name}/{split}/{model_name}{'_debug' if debug else ''}.csv"
	predictions = load_dataset(pred_frn)

	# evaluate
	acc = evaluate_acc(predictions, feature)
	print(f"Model: {model_name}\nAccuracy: {acc}")