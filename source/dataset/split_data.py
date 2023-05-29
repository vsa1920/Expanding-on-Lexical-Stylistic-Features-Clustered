'''Split the extracted pairs into (train,) val and test.'''
import os
if os.getcwd().endswith('dataset'):
	os.chdir('../..')
import sys
import csv
import random
from dataset.utils import load_dataset

def write_pairs(dataset, fwn):
	with open(fwn, 'w') as fw:
		writer = csv.DictWriter(fw, fieldnames=dataset[0].keys())
		writer.writeheader()
		for row in dataset:
			writer.writerow(row)

if __name__ == "__main__":
	# config

	# modify these when adding a new dataset
	source_split = ["all", "idiom", "metaphor"][-2]
	feature = ["complexity", "formality", "intensity", "figurativeness"][-1]
	dataset_name = ["SimpleWikipedia", "GYAFC", "IMPLI"][-1]
	# no need to modify
	splits = ["train", "val", "test"][1:] # when the train set is not needed, we take [1:]
	split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}

	# load the cleaned dataset
	dataset_frn = f"data/{feature}/{dataset_name}/{source_split}.csv"
	dataset = load_dataset(dataset_frn)

	# split into sets
	random.shuffle(dataset)
	total_ratio = sum([v for k,v in split_ratio.items() if k in splits])
	normalized_ratio = {k: v/total_ratio for k,v in split_ratio.items() if k in splits}

	n_total_examples = len(dataset)
	for split in splits:
		n_examples = int(n_total_examples*normalized_ratio[split])
		print(f"{split}: {n_examples} examples.")
		split_dataset = dataset[:n_examples]
		dataset = dataset[n_examples:]

		fwn = f"data/{feature}/{dataset_name}/{source_split}_{split}.csv"
		write_pairs(split_dataset, fwn)



