'''Extract pairs of texts from a raw dataset.'''
import os
if os.getcwd().endswith('dataset'):
	os.chdir('../..')
import sys
import csv
import random
from dataset.utils import load_dataset, PREDICTED_CLASS_MAP

def is_pair_clean(row, feature, dataset_name, thresh=0.8):
	'''Check if a pair is clean. The rules should be designed based on each dataset.
	:param
		row: a row of the dataset. The format depends on the dataset.
		feature (str): the target feature. One of ["complexity", "formality", "intensity", "figurativeness"].
		dataset_name (str): the name of the dataset.
		thresh (float): the threshold for the human agreement/confidence in order for a pair to be considered clean. Only used for certain datasets.
	:return
		True if the pair is clean, False otherwise.
	'''
	if feature == "complexity":
		if dataset_name == "SimplePPDB":
			gold = row['gold'][len("turkers="):].strip('"')
			if gold == "NA":
				return False
			majority = int(row['majority'])
			total = int(row['total'])
			if majority / total < thresh: # check for human agreement
				return False
		elif dataset_name == "SimpleWikipedia":
			simple_text, complex_text = row["simple"], row["complex"]
			simple_tokens, complex_tokens = simple_text.split(), complex_text.split()
			if set(simple_tokens) == set(complex_tokens): # they have the same set of tokens
				return False
			if set(simple_tokens).issubset(complex_tokens) or set(complex_tokens).issubset(simple_tokens): # one is a subset of the other
				return False
			return True
	elif feature == "formality":
		if dataset_name == "StylePPDB":
			if row["note"] == "NA":
				return False
			if float(row["Confidence"]) < thresh: # check for human confidence
				return False
		elif dataset_name == "GYAFC":
			# every pair is human-annotated, so there's no need to check
			return True
	elif feature == "figurativeness":
		if dataset_name == "IMPLI":
			# every pair is human-annotated, so there's no need to check
			return True
	return True

def extract_clean_pairs(dataset, feature, dataset_name):
	'''Extract clean pairs from a raw dataset.
	:param
		dataset (list): a list of rows of the dataset. The format depends on the dataset.
		feature (str): the target feature. One of ["complexity", "formality", "intensity", "figurativeness"].
		dataset_name (str): the name of the dataset.
	:return
		clean_dataset (list): a list of clean rows of the dataset. Each row should be in the format of {"0": text0, "1": text1, "gold_{predicted_class}": 0/1}.
	'''
	predicted_class = PREDICTED_CLASS_MAP[feature]
	clean_dataset = []
	for row in dataset:
		if is_pair_clean(row, feature=feature, dataset_name=dataset_name):
			if feature == "complexity":
				if dataset_name == "SimplePPDB":
					gold = row['gold'][len("turkers="):].strip('"')
					new_row = {"0": row["x"],
					           "1": row["y"],
					           f"gold_{predicted_class}": 0 if gold == row["x"] else 1}
				elif dataset_name == "SimpleWikipedia":
					simple, complex = row["simple"], row["complex"]
					answer_id = random.sample([0,1], 1)[0]
					if answer_id == 0:
						new_row = {"0": simple,
						           "1": complex,
						           f"gold_{predicted_class}": answer_id}
					else:
						new_row = {"0": complex,
						           "1": simple,
						           f"gold_{predicted_class}": answer_id}
			elif feature == "formality":
				if dataset_name in ["StylePPDB", "GYAFC"]:
					informal, formal = row["informal"], row["formal"]
					answer_id = random.sample([0,1], 1)[0]
					if answer_id == 0:
						new_row = {"0": informal,
						           "1": formal,
						           f"gold_{predicted_class}": answer_id}
					else:
						new_row = {"0": formal,
						           "1": informal,
						           f"gold_{predicted_class}": answer_id}
			elif feature == "figurativeness":
				if dataset_name == "IMPLI":
					literal, figurative = row["literal"], row["figurative"]
					answer_id = random.sample([0,1], 1)[0]
					if answer_id == 0:
						new_row = {"0": literal,
						           "1": figurative,
						           f"gold_{predicted_class}": answer_id}
					else:
						new_row = {"0": figurative,
						           "1": literal,
						           f"gold_{predicted_class}": answer_id}
			else:
				raise ValueError(f"Unknown feature: {feature}.")
			clean_dataset.append(new_row)
	return clean_dataset

def write_pairs(dataset, fwn):
	with open(fwn, 'w') as fw:
		writer = csv.DictWriter(fw, fieldnames=dataset[0].keys())
		writer.writeheader()
		for row in dataset:
			writer.writerow(row)

if __name__ == "__main__":
	# config
	# modify these when adding a new dataset
	feature = ["complexity", "formality", "figurativeness"][-1]
	dataset_name = ["SimpleWikipedia", "GYAFC", "IMPLI"][-1]
	raw_fn = ["all", "idiom", "metaphor"][-1]

	# load raw dataset
	dataset_frn = f"data/{feature}/{dataset_name}/raw/{raw_fn}.tsv"
	dataset = load_dataset(dataset_frn)

	# extract clean pairs of texts from raw dataset
	clean_dataset = extract_clean_pairs(dataset, feature=feature, dataset_name=dataset_name)

	# write clean dataset to file
	fwn = f"data/{feature}/{dataset_name}/all.csv"
	write_pairs(clean_dataset, fwn)



