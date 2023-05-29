import csv

ALL_FEATURES = ['complexity', 'formality', 'figurativeness']

DATASET_MAP = {
	"complexity": {"short": "SimplePPDB",
	               "long": "SimpleWikipedia"
	               },
	"figurativeness": {"short": None,
	                   "long": "IMPLI"
	                   },
	"formality": {"short": "StylePPDB",
	              "long": "GYAFC"
	              }
}

# the target class name in our intrinsic evaluation task for each feature
PREDICTED_CLASS_MAP = {
	"complexity": "simple",
	"formality": "informal",
	"figurativeness": "literal",
	"intensity": "mild"
}

# majority baseline performance
MAJ_BASELINE_PERF = {
	"complexity": {"short": 55.1,
	               "long": 50.6
	               },
	"figurativeness": {"short": "-",
	                   "long": 51.4
	                   },
	"formality": {"short": 51.2,
	              "long": 51.8
	              }
}



def load_dataset(dataset_frn):
	if dataset_frn.endswith('.csv'):
		delimiter = ','
	elif dataset_frn.endswith('.tsv'):
		delimiter = '\t'
	else:
		raise ValueError("Unknown dataset file format.")

	with open(dataset_frn, 'r', encoding='utf-8') as dataset_fr:
		reader = csv.DictReader(dataset_fr, delimiter=delimiter)
		rows = []
		for row in reader:
			rows.append(row)
		return rows