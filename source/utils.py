import json
from nltk import Tree
import pickle
from time import time
from transformers import AutoModel, AutoTokenizer
import spacy
import numpy as np
from gensim.models import KeyedVectors
import torch

POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]

POS_TAG_MAP = {
	"ADJ": "ADJ",
	"ADP": "ADP",
	"ADV": "ADV",
	"AUX": "OTHER",
	"CCONJ": "CONJ",
	"DET": "DET",
	"INTJ": "OTHER",
	"NOUN": "NOUN/PROPN/PRON",
	"NUM": "OTHER",
    "PART": "OTHER",
    "PRON": "NOUN/PROPN/PRON",
    "PROPN": "NOUN/PROPN/PRON",
    "PUNCT": "OTHER",
    "SCONJ": "CONJ",
    "SYM": "OTHER",
    "VERB": "VERB",
    "X": "OTHER",
    "SPACE": "OTHER"
}


PREDICTED_CLASS = {
	"complexity": "simple",
	"formality": "informal",
	"intensity": "mild",
	"figurativeness": "literal"
}

def _get_ranks(x: torch.Tensor) -> torch.Tensor:
	tmp = x.argsort()
	ranks = torch.zeros_like(tmp)
	ranks[tmp] = torch.arange(len(x), device=x.device)
	return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
	"""Compute correlation between 2 1-D vectors
	Args:
		x: Shape (N, )
		y: Shape (N, )
	"""
	x_rank = _get_ranks(x)
	y_rank = _get_ranks(y)

	n = x.size(0)
	upper = 6 * torch.sum((x_rank - y_rank).pow(2))
	down = n * (n ** 2 - 1.0)
	return 1.0 - (upper / down)

def load_spacy(model: str):
	nlp = spacy.load(model)
	for name in ["lemmatizer", "ner"]:
		nlp.remove_pipe(name)
	return nlp


def timer_func(func):
	# This function shows the execution time of the function object passed
	# Credits: https://www.geeksforgeeks.org/timing-functions-with-decorators-python/
	def wrap_func(*args, **kwargs):
		t1 = time()
		result = func(*args, **kwargs)
		t2 = time()
		print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
		return result

	return wrap_func


def load_json(path) -> dict:
	"""Loads a JSON as a dict"""
	with open(path, "r") as fin:
		data = json.load(fin)
		return data


def save_json(data: dict, path, mode="w") -> None:
	"""Saves a dict as a JSON"""
	with open(path, mode) as fout:
		json.dump(data, fout, ensure_ascii=False, indent=2)


def load_spacy(model: str):
	nlp = spacy.load(model)
	for name in ["lemmatizer", "ner"]:
		nlp.remove_pipe(name)
	return nlp


def save_pkl(data, path):
	with open(path, "ab") as fout:
		pickle.dump(data, fout)


def load_pkl(path):
	with open(path, "rb") as fin:
		return pickle.load(fin)


def remove_dupes(iterable):
	"""Removes duplicates from an iterable"""
	checked = []
	for n in iterable:
		if n not in checked:
			checked.append(n)
	return checked


def _to_nltk_tree(node):
	"""
	Converts a spacy parse tree into nltk Tree (for visualization purposes w/o using displacy)
	Credits: https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
	"""
	tok_format = lambda tok: "_".join([tok.orth_, tok.tag_, tok.dep_])
	if node.n_lefts + node.n_rights > 0:
		return Tree(tok_format(node), [_to_nltk_tree(child) for child in node.children])
	else:
		return tok_format(node)


def tree(root):
	"""Print a NLTK style parse tree given the root"""
	_to_nltk_tree(root).pretty_print()

def load_static_embedding_model(frn):
	'''Load a static word embedding model.'''
	assert frn.endswith('vec')
	model = KeyedVectors.load_word2vec_format(frn, binary=False)
	return model

def get_static_embedding(model, word: str) -> np.ndarray:
	'''Get a static word embedding from a gensim KeyedVectors model.
	:param model: a gensim KeyedVectors model
	:param word: a word

	:return: a numpy array of the word embedding
	'''
	zero_vector = np.zeros(model.vector_size)
	try:
		vector = model.get_vector(word)
	except KeyError:
		vector = zero_vector
	return vector

def load_model_and_tokenizer(LM_name: str, device):
	if "bert" in LM_name:
		LM = AutoModel.from_pretrained(LM_name).to(device)
		tokenizer = AutoTokenizer.from_pretrained(LM_name, do_lower_case=True, add_prefix_space=True)
	else:
		LM_type = "glove" if "glove" in LM_name else "fasttext"
		LM_frn = f"data/word_embeddings/{LM_type}/{LM_name}.vec"
		LM = load_static_embedding_model(LM_frn)
		tokenizer = None
	return LM, tokenizer



