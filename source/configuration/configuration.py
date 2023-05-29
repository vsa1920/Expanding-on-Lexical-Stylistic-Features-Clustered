'''
Model configuration.
Adapted from http://blender.cs.illinois.edu/software/oneie.
'''

import copy
import json
import os
from typing import Dict

class Config(object):
	def __init__(self, **kwargs):
		self.features = kwargs.get('features', "[]") # features that the model supports

		# each feature has its own LM, layers, and seeds
		self.LM_names = kwargs.get('LM_names', {}) # the underlying LM to generate embeddings
		self.seeds_fns = kwargs.get('seeds_fns', {}) # the file containing the seeds
		self.layers = kwargs.get('layers', {}) # the LM layer to extract embeddings from (e.g. -1)

		self.pooling = kwargs.get('pooling', 'mean') # the pooling method to generate feature values for a text out of token embeddings

		self.combine_ngram = kwargs.get('combine_ngram', False) # whether to combine with Eric's n-gram features
		self.freq_fn = kwargs.get('freq_fn', "") # the file containing the Google ngram frequency features
		self.layeragg = kwargs.get('layeragg', False) # aggragate embeddings from the first to the current layer
		self.isotropy = kwargs.get('isotropy', None) # anisotropy reduction strategy

	@classmethod
	def from_dict(cls, dict_obj):
		"""Creates a Config object from a dictionary.
		Args:
			dict_obj (Dict[str, Any]): a dict where keys are
		"""
		config = cls()
		for k, v in dict_obj.items():
			setattr(config, k, v)
		return config

	@classmethod
	def from_json_file(cls, path):
		with open(path, 'r', encoding='utf-8') as r:
			return cls.from_dict(json.load(r))

	def to_dict(self):
		output = copy.deepcopy(self.__dict__)
		return output

	def save_config(self, path):
		"""Save a configuration object to a file.
		:param path (str): path to the output file or its parent directory.
		"""
		if os.path.isdir(path):
			path = os.path.join(path, 'config.json')
		print('Save config to {}'.format(path))
		with open(path, 'w', encoding='utf-8') as w:
			w.write(json.dumps(self.to_dict(), indent=2, sort_keys=True))