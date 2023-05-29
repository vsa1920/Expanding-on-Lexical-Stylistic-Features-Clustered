import os
if "/shared/lyuqing/" in os.getcwd():
	os.environ["HF_HOME"] = "/shared/lyuqing/huggingface_cache"
if os.getcwd().endswith('model'):
	os.chdir('../..')
import sys
sys.path.append('source')
import csv
import torch
torch.manual_seed(42)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import pickle
from torch.nn.functional import cosine_similarity
import random
random.seed(42)
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from dataset.utils import PREDICTED_CLASS_MAP
from utils import POS_TAGS, load_spacy, load_model_and_tokenizer, get_static_embedding, spearman_correlation
from configuration.configuration import Config

class GoldModel:
	def __init__(self):
		super(GoldModel, self).__init__()

	def compare_feature_batch(self, examples, feature):
		preds = []
		for example in examples:
			if feature == "complexity":
				preds.append(example["gold_simple"])
			elif feature == "formality":
				preds.append(example["gold_informal"])
			elif feature == "intensity":
				preds.append(example["gold_mild"])
		return preds

class FreqFeaturizer:
	def __init__(self, config):
		super(FreqFeaturizer, self).__init__()
		self.features = eval(config.features)
		self.freq_counts = self.load_freq_counts(config.freq_fn)
		self.pooling = config.pooling
		self.nlp = load_spacy("en_core_web_md")

	def load_freq_counts(self, frn):
		ngram_counts = {}
		assert os.path.exists(frn), f"File {frn} does not exist. Current working directory: {os.getcwd()}."
		with open(frn, "r") as fr:
			reader = csv.DictReader(fr, delimiter="\t")
			for row in reader:
				ngram, count = row["word"].lower(), int(row["count"])
				ngram_counts[ngram] = count
		return ngram_counts

	def parse_batch(self, texts):
		'''Tokenize and do POS taggging on a list of texts'''
		docs = list(self.nlp.pipe(texts))
		tokens_for_docs = []
		pos_tags_for_docs = []
		for doc in docs:
			tokens = [token.text for token in doc]
			pos_tags = [token.pos_ for token in doc]
			tokens_for_docs.append(tokens)
			pos_tags_for_docs.append(pos_tags)
		return tokens_for_docs, pos_tags_for_docs

	def get_freqs(self, tokens: list[str]) -> torch.Tensor:
		''' Get the frequencies for a list of tokens.
		:param tokens: a list of tokens.
		:return freqs: a tensor of shape [n_tokens, 1] representing the frequency for each token.
		'''
		freqs = []
		for token in tokens:
			if token.lower() in self.freq_counts:
				freqs.append(self.freq_counts[token.lower()])
			else:
				freqs.append(0)
		freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
		return freqs

	def predict_feature_batch(self, texts, feature):
		''' Predict the frequency for a batch of texts.

		:param texts: a list of texts, each of which is a string.
		:param feature: the target feature to predict, which can be "complexity", "formality", "intensity", or "figurativeness".

		:return: features_for_texts: a tensor of shape [n_texts, n_dim] representing the feature scores for each text, where n_dim is the dimension of the feature.
		'''

		assert feature in self.features, f"Feature {feature} is not in the list of features {self.features}."

		# tokenize and POS tag the texts
		tokens_for_texts, pos_tags_for_texts = self.parse_batch(texts)
		features_for_texts = []

		if self.pooling == "mean":
			# get the average token embeddings as sentence embedding
			for tokens_for_text in tokens_for_texts:
				token_freqs = self.get_freqs(tokens_for_text) # shape: [n_tokens, 1]
				mean_text_freq = torch.mean(token_freqs, dim=0) # shape: [1]
				features_for_text = [mean_text_freq]
				features_for_text = torch.stack(features_for_text)
				features_for_texts.append(features_for_text)
			features_for_texts = torch.stack(features_for_texts)

		elif self.pooling == "max":
			# get the raw token embeddings
			for tokens_for_text in tokens_for_texts:
				token_freqs = self.get_freqs(tokens_for_text) # shape: [n_tokens, 1]
				max_text_freq = torch.max(token_freqs, dim=0)[0] # shape: [1]
				features_for_text = [max_text_freq]
				features_for_text = torch.stack(features_for_text)
				features_for_texts.append(features_for_text)
			features_for_texts = torch.stack(features_for_texts)

		return features_for_texts

	def compare_feature_batch(self, examples, feature):
		'''Compare the feature (complexity, formality, intensity, figurativeness) of two pieces of texts in a batch of examples.
		If text0 is simple/informal/mild/literal, return 0.
		If text1 is simple/informal/mild/literal, return 1.
		Else, return a random index.
		'''
		assert feature in self.features, f"Feature {feature} is not in the list of features {self.features}."

		text0s, text1s = [], []
		for i, example in enumerate(examples):
			text0, text1 = example["0"], example["1"]
			text0s.append(text0)
			text1s.append(text1)

		# predict the frequency of texts
		text0s_fvalue = self.predict_feature_batch(text0s, feature)
		text1s_fvalue = self.predict_feature_batch(text1s, feature)
		preds = []
		for i in range(len(text0s)):
			text0, text1 = text0s[i], text1s[i]
			text0_fvalue, text1_fvalue = text0s_fvalue[i], text1s_fvalue[i]
			# higher frequency means more simple/informal/mild/literal
			if text0_fvalue > text1_fvalue:
				preds.append(0)
			elif text1_fvalue > text0_fvalue:
				preds.append(1)
			else:
				print(f"Same {feature} between {text0} and {text1}: {text0_fvalue}.")
				random_label = random.sample([0,1], 1)[0]
				preds.append(random_label)
		return preds

class LexicalFeaturizer:

	def __init__(self, config):
		super(LexicalFeaturizer, self).__init__()
		self.features = eval(config.features)
		self.layeragg = config.layeragg

		# isotropy reduction strategy
		self.isotropy = config.isotropy
		assert self.isotropy in [None, "normalized", "abtt", "rank"], f"Isotropy improvement method {self.isotropy} is not supported."
		if self.isotropy:
			self.isotropy_vecs = {feature:{"mean": None,
			                               "std": None,
			                               "pcs": None}
			                      for feature in self.features}
			self.isotropy_vec_fns = {}

		if not self.features: # user did not specify features
			self.features = config.LM_names.keys()

		self.LM_names, self.seeds_fns, self.layers, self.dvec_fns = {}, {}, {}, {}
		for feature in self.features:
			self.LM_names[feature] = config.LM_names[feature]
			self.seeds_fns[feature] = config.seeds_fns[feature]
			self.layers[feature] = config.layers[feature]
			seeds_fn = self.seeds_fns[feature]
			assert seeds_fn, f"Please specify the seeds file for {feature}."
			seeds_id = seeds_fn.split('/')[-1].split('.')[0].split('_')[1]
			self.dvec_fns[feature] = f"data/{feature}/dvecs/{self.LM_names[feature]}_{self.layers[feature]}{'_layeragg' if self.layeragg else ''}_seeds{seeds_id}{f'_{self.isotropy}' if self.isotropy else ''}_dvec.pkl"
			if self.isotropy:
				self.isotropy_vec_fns[feature] = f"data/{feature}/dvecs/{self.LM_names[feature]}_{self.layers[feature]}{'_layeragg' if self.layeragg else ''}_seeds{seeds_id}{f'_{self.isotropy}' if self.isotropy else ''}_isovec.pkl"

		self.dvecs = {feature: None for feature in self.features}
		self.LMs = {feature: None for feature in self.features}
		self.tokenizers = {feature: None for feature in self.features}

		self.pooling = config.pooling
		self.nlp = load_spacy("en_core_web_md")

	def load_LM(self):
		'''Load the language model and tokenizer.'''
		for feature in self.features:
			if self.LM_names[feature]:
				self.LMs[feature], self.tokenizers[feature] = load_model_and_tokenizer(self.LM_names[feature], device)

	def get_embeddings(self, texts, feature) -> list[torch.Tensor]:
		'''Get the embeddings of a list of texts.
		Args:
			texts (list): a list of texts, each of which is a list of tokens
		Returns:
			texts_embeddings (list): a list of tensors, each of shape (n_tokens, embedding_dim)
		'''
		LM = self.LMs[feature]
		tokenizer = self.tokenizers[feature]
		layer = self.layers[feature]

		if tokenizer: # Contextualized models
			# tokenize the texts
			inputs = tokenizer(texts, is_split_into_words=True, padding=True, return_tensors="pt", truncation=True).to(device)
			# get the embeddings
			LM.eval()
			with torch.no_grad():
				# process one batch at a time
				texts_embeddings = []
				outputs = LM(**inputs, output_hidden_states=True)
				# get the embeddings of the specified layer
				# print(f"Number of layers in {self.LM_names[feature]}: {len(outputs.hidden_states)}")
				if self.layeragg: # average embeddings from the first to the current layer
					if layer == -1:
						layer = len(outputs.hidden_states) - 1
					token_embeddings = torch.stack(outputs.hidden_states[:layer+1], dim=0).mean(dim=0)
				else:
					token_embeddings = outputs.hidden_states[layer]
				# for each text, get the averaged token embeddings
				for text_id, token_embedding in enumerate(token_embeddings):
					# align the token embeddings with the pre-split tokens
					# if a pre-split token is then split into multiple wordpieces, average the embeddings of the wordpieces
					aligned_token_embedding = []
					transformer_to_presplit_word_id_mapping = inputs.word_ids(batch_index=text_id)
					previous_word_idx = -1
					current_token_wordpieces_embeddings = []

					for transformer_token_id, presplit_word_id in enumerate(transformer_to_presplit_word_id_mapping):
						if presplit_word_id is None:
							continue
						elif presplit_word_id != previous_word_idx:  # new word
							# add the previous token
							if current_token_wordpieces_embeddings:
								current_token_wordpieces_embeddings = torch.stack(current_token_wordpieces_embeddings)
								aligned_token_embedding.append(torch.mean(current_token_wordpieces_embeddings, dim=0))
							# start a new token
							current_token_wordpieces_embeddings = [token_embedding[transformer_token_id]]
							if presplit_word_id != previous_word_idx + 1: # new token id is not continuous
								# append zero embeddings for missing tokens
								for _ in range(presplit_word_id - previous_word_idx - 1):
									aligned_token_embedding.append(torch.zeros(LM.config.hidden_size).to(device))
						else: # same word
							current_token_wordpieces_embeddings.append(token_embedding[transformer_token_id])
						previous_word_idx = presplit_word_id

					# add the last token
					current_token_wordpieces_embeddings = torch.stack(current_token_wordpieces_embeddings)
					aligned_token_embedding.append(torch.mean(current_token_wordpieces_embeddings, dim=0))
					n_tokens = len(texts[text_id])
					for _ in range(n_tokens - previous_word_idx - 1):
						aligned_token_embedding.append(torch.zeros(LM.config.hidden_size).to(device))
					aligned_token_embedding = torch.stack(aligned_token_embedding)

					# add to the list of sentence embeddings
					texts_embeddings.append(aligned_token_embedding)

				# texts_embeddings = torch.stack(texts_embeddings)

		else: # Static models
			# get the embeddings
			texts_embeddings = []
			for text in texts:
				text_embedding = []
				for token in text:
					token_embedding = torch.from_numpy(get_static_embedding(LM, token).copy()).to(device)
					text_embedding.append(token_embedding)
				text_embedding = torch.stack(text_embedding)
				texts_embeddings.append(text_embedding)
			# texts_embeddings = torch.stack(texts_embeddings)
		return texts_embeddings

	def parse_batch(self, texts):
		'''Tokenize and do POS taggging on a list of texts'''
		docs = list(self.nlp.pipe(texts))
		tokens_for_docs = []
		pos_tags_for_docs = []
		for doc in docs:
			tokens = [token.text for token in doc]
			pos_tags = [token.pos_ for token in doc]
			tokens_for_docs.append(tokens)
			pos_tags_for_docs.append(pos_tags)
		return tokens_for_docs, pos_tags_for_docs

	def generate_dvecs(self):
		'''Generate the dvec representing the feature from the seeds, and save it to the self.dvec_fn file.'''

		for feature in self.features:
			already_generated = True
			if not os.path.exists(self.dvec_fns[feature]):
				already_generated = False
			if not (self.isotropy and os.path.exists(self.isotropy_vec_fns[feature])):
				already_generated = False
			if already_generated:
				self.load_dvecs()
				continue

			seeds_fn = self.seeds_fns[feature]

			with open(seeds_fn, 'r') as seeds_fr:
				reader = csv.DictReader(seeds_fr)
				fless_texts, fmore_texts = [], []
				for row in reader:
					# fless means the feature is less strong (e.g., simple, informal)
					# fmore means the feature is more strong (e.g., complex, formal)
					fless_text, fmore_text = row["0"], row["1"]
					fless_texts.append(fless_text)
					fmore_texts.append(fmore_text)

				seed_texts = [f"{fmore_text} - {fless_text}" for fless_text, fmore_text in zip(fless_texts, fmore_texts)]
				tokens_for_fless_texts, _ = self.parse_batch(fless_texts)
				tokens_for_fmore_texts, _ = self.parse_batch(fmore_texts)

				fless_embs = self.get_embeddings(tokens_for_fless_texts, feature=feature) # a list of tensors, each of shape (n_tokens, emb_dim)
				# compute the average of the embeddings of the tokens in each text
				fless_embs_mean = []
				for emb in fless_embs:
					mean_emb = torch.mean(emb, dim=0)
					fless_embs_mean.append(mean_emb)
				fless_embs_mean = torch.stack(fless_embs_mean)

				fmore_embs = self.get_embeddings(tokens_for_fmore_texts, feature=feature)
				# compute the average of the embeddings of the tokens in each text
				fmore_embs_mean = []
				for emb in fmore_embs:
					mean_emb = torch.mean(emb, dim=0)
					fmore_embs_mean.append(mean_emb)
				fmore_embs_mean = torch.stack(fmore_embs_mean)

			# subtract each simple embedding from each complex embedding
			diff_embs = fmore_embs_mean - fless_embs_mean
			# visualize the embeddings
			# self.visualize_vecs(diff_embs, seed_texts)

			# get dvec by taking the average of the differences
			dvec = torch.mean(diff_embs, dim=0)
			self.dvecs[feature] = dvec
			# save the dvec to the file
			dvec_path = "/".join(self.dvec_fns[feature].split("/")[:-1])
			if not os.path.exists(dvec_path):
				os.makedirs(dvec_path)
			with open(self.dvec_fns[feature], 'wb') as dvec_fw:
				pickle.dump(dvec, dvec_fw)

			# compute isotropy vecs
			if self.isotropy:
				# stack fless_embs_mean and fmore_embs_mean
				all_embs = torch.cat((fless_embs_mean, fmore_embs_mean), dim=0) # (n_texts, emb_dim)
				isotropy_vecs = self.compute_mean_std_pcs(all_embs)
				self.isotropy_vecs[feature]["mean"] = isotropy_vecs["mean"]
				self.isotropy_vecs[feature]["std"] = isotropy_vecs["std"]
				self.isotropy_vecs[feature]["pcs"] = isotropy_vecs["pcs"]
				# save the isotropy vecs to the file
				isotropy_fwn = self.isotropy_vec_fns[feature]
				with open(isotropy_fwn, 'wb') as isotropy_fw:
					pickle.dump(self.isotropy_vecs, isotropy_fw)

	def visualize_vecs(self, vecs, seed_texts):
		'''Visualize the vectors generated from the seed_texts.'''
		pca = PCA(n_components=2)
		pipe = Pipeline([('normalizer', Normalizer()), ('pca', pca)])
		vecs_t = pipe.fit_transform(vecs)
		# plot the vectors, with the seed texts as labels
		plt.figure(figsize=(10, 10))
		plt.xlim(-1, 1)
		plt.ylim(-1, 1)
		plt.scatter(vecs_t[:, 0], vecs_t[:, 1])
		for i, txt in enumerate(seed_texts):
			plt.annotate(txt, (vecs_t[i, 0], vecs_t[i, 1]))
		# plot = plt.scatter(vecs_t[:, 0], vecs_t[:, 1],
		# plt.legend(handles=plot.legend_elements()[0], labels=seed_texts)
		plt.show()

	def load_dvecs(self):
		for feature in self.features:
			with open(self.dvec_fns[feature], 'rb') as dvec_fr:
				dvec = pickle.load(dvec_fr)
			self.dvecs[feature] = dvec.to(device)

			# load the isotropy vecs
			if self.isotropy:
				isotropy_frn = self.isotropy_vec_fns[feature]
				with open(isotropy_frn, 'rb') as isotropy_fr:
					self.isotropy_vecs = pickle.load(isotropy_fr)
				self.isotropy_vecs[feature]["mean"] = self.isotropy_vecs[feature]["mean"].to(device)
				self.isotropy_vecs[feature]["std"] = self.isotropy_vecs[feature]["std"].to(device)
				self.isotropy_vecs[feature]["pcs"] = self.isotropy_vecs[feature]["pcs"].to(device)

	def similarity(self, emb1, emb2, feature):
		'''Compute the generalized cosine similarity between two embeddings.
		:param emb1: a tensor of shape (n_dim)
		:param emb2: a tensor of shape (n_dim)

		:return: a float representing the similarity score.
		'''

		if not self.isotropy:
			return cosine_similarity(emb1, emb2, dim=0)

		if self.isotropy == "rank":
			corr = spearman_correlation(emb1, emb2)
			# convert to tensor
			corr = torch.tensor(corr, dtype=torch.float32, device=device)
			return corr

		emb1_mean_rm = (emb1 - self.isotropy_vecs[feature]["mean"])
		emb2_mean_rm = (emb2 - self.isotropy_vecs[feature]["mean"])

		if self.isotropy == "normalized":
			emb1_normalized = emb1_mean_rm / self.isotropy_vecs[feature]["std"]
			emb2_normalized = emb2_mean_rm / self.isotropy_vecs[feature]["std"]
			return cosine_similarity(emb1_normalized, emb2_normalized, dim=0)

		if self.isotropy == "abtt":
			# removing the pcs from the embeddings
			pcs = self.isotropy_vecs[feature]["pcs"]
			# Compute dot products of pcs with emb1_mean_rm and emb2_mean_rm
			pc_dot1 = pcs @ emb1_mean_rm
			pc_dot2 = pcs @ emb2_mean_rm
			rm_term1 = (pc_dot1[..., None] * pcs).sum(axis=0)
			rm_term2 = (pc_dot2[..., None] * pcs).sum(axis=0)
			emb1_abtt = emb1_mean_rm - rm_term1
			emb2_abtt = emb2_mean_rm - rm_term2
			return cosine_similarity(emb1_abtt, emb2_abtt, dim=0)

		raise ValueError("isotropy must be one of 'rank', 'normalized', 'abtt', or None.")

	def predict_feature_batch(self, texts, feature):
		''' Predict the target feature for a batch of texts.

		:param texts: a list of texts, each of which is a string.
		:param feature: the target feature to predict, which can be "complexity", "formality", "intensity", or "figurativeness".

		:return: features_for_texts: a tensor of shape [n_texts, n_dim] representing the feature scores for each text, where n_dim is the dimension of the feature.
		'''

		try:
			assert self.dvecs[feature] is not None
		except:
			raise AssertionError("dvec is not loaded.")

		# tokenize and POS tag the texts
		tokens_for_texts, pos_tags_for_texts = self.parse_batch(texts)
		features_for_texts = []

		if self.pooling == "mean":
			# get the average token embeddings as sentence embedding
			for tokens_for_text in tokens_for_texts:
				token_embs = self.get_embeddings([tokens_for_text], feature=feature)[0] # shape: [n_tokens, emb_dim]
				mean_text_emb = torch.mean(token_embs, dim=0) # shape: [emb_dim]
				# compute the generalized similarity between the text embedding and the dvec
				# cos_sim = cosine_similarity(mean_text_emb, self.dvecs[feature].unsqueeze(0)).squeeze(0)
				cos_sim = self.similarity(mean_text_emb, self.dvecs[feature], feature)
				features_for_text = [cos_sim]
				features_for_text = torch.stack(features_for_text)
				features_for_texts.append(features_for_text)
			features_for_texts = torch.stack(features_for_texts)

		elif self.pooling == "max":
			# get the raw token embeddings
			for tokens_for_text in tokens_for_texts:
				# get the token embedding for the current text
				token_embs = self.get_embeddings([tokens_for_text], feature=feature)[0] # shape: [n_tokens, emb_dim]
				# compute the generalized similarity between each token embedding in the text and the dvec
				# cos_sim = cosine_similarity(token_embs, self.dvecs[feature].unsqueeze(0)).squeeze(0) # shape: [n_tokens]
				cos_sim = torch.stack([self.similarity(token_emb, self.dvecs[feature], feature) for token_emb in token_embs]) # shape: [n_tokens]
				# get the maximum cosine similarity
				max_cos_sim = torch.max(cos_sim)
				# get the token with maximum cosine similarity
				# max_cos_sim_token = tokens[torch.argmax(cos_sim)]
				# print(max_cos_sim_token, max_cos_sim)
				features_for_text = [max_cos_sim]
				features_for_text = torch.stack(features_for_text)
				features_for_texts.append(features_for_text)
			features_for_texts = torch.stack(features_for_texts)

		elif self.pooling == "byPOS":
			# get the average token embeddings as sentence embedding
			for text_id, tokens_for_text in enumerate(tokens_for_texts):
				token_embs = self.get_embeddings([tokens_for_text], feature=feature)[0] # shape: [n_tokens, emb_dim]
				# compute the generalized similarity between the text embedding and the dvec
				# cos_sim = cosine_similarity(token_embs, self.dvecs[feature].unsqueeze(0)).squeeze(0) # shape: [n_tokens]
				cos_sim = torch.stack([self.similarity(token_emb, self.dvecs[feature], feature) for token_emb in token_embs]) # shape: [n_tokens]
				# aggregate the cosine similarity by POS tags
				pos_tags_for_text = pos_tags_for_texts[text_id]
				features_for_text = []
				for pos_tag in POS_TAGS:
					# get the cosine similarity for the current POS tag
					mask = [pt == pos_tag for pt in pos_tags_for_text]
					pos_tag_cos_sim = torch.mean(cos_sim[mask])
					features_for_text.append(pos_tag_cos_sim)
				features_for_text = torch.stack(features_for_text)
				# replace nan values
				features_for_text = torch.nan_to_num(features_for_text, nan=0.0)
				features_for_texts.append(features_for_text)
			features_for_texts = torch.stack(features_for_texts)

		return features_for_texts

	def compare_feature_batch(self, examples, feature):
		'''Compare the feature (complexity, formality) of two pieces of texts in a batch of examples.
		If text0 is simpler/more informal, return 0.
		If text1 is simpler/more informal, return 1.
		Else, return a random index.
		'''
		text0s, text1s = [], []
		for i, example in enumerate(examples):
			text0, text1 = example["0"], example["1"]
			text0s.append(text0)
			text1s.append(text1)

		text0s_fvalue = self.predict_feature_batch(text0s, feature)
		text1s_fvalue = self.predict_feature_batch(text1s, feature)
		preds = []
		for i in range(len(text0s)):
			text0_fvalue, text1_fvalue = text0s_fvalue[i], text1s_fvalue[i]
			if text0_fvalue < text1_fvalue:
				preds.append(0)
			elif text1_fvalue < text0_fvalue:
				preds.append(1)
			else:
				text0, text1 = text0s[i], text1s[i]
				print(f"Same {feature} between {text0} and {text1}: {text0_fvalue}.")
				random_label = random.sample([0,1], 1)[0]
				preds.append(random_label)
		return preds

	def vectorize(self, text:str) -> np.ndarray:
		'''Get the vector representation of a text.
		'''
		feature_values = [self.predict_feature_batch([text], feature).squeeze(0) for feature in self.features]
		concatenated_features = torch.cat(feature_values, dim=0).cpu().numpy()
		return concatenated_features

	def compute_mean_std_pcs(self, embeddings:torch.Tensor) -> dict:
		'''Compute the mean, the standard deviation, and the principal components of embeddings (for isotropy reduction).

		:param embeddings: embeddings of the texts, a Tensor of shape (n_texts, n_dim)

		:return: dict of mean (n_dim), standard deviation (n_dim), and principal components (n_components, n_dim)
		'''
		# compute the mean
		means = torch.mean(embeddings, dim=0)
		# compute the standard deviation
		stds = torch.std(embeddings, dim=0)

		# normalize the layer embedding
		normalized_embeddings = embeddings - means
		# move to cpu
		normalized_embeddings = normalized_embeddings.cpu().numpy()
		# compute the principal components
		n_components = int(embeddings.shape[1] / 100) # removing n_dim/100 components
		pca = PCA(n_components=n_components)
		pca.fit(normalized_embeddings)
		pcs = pca.components_
		# convert to torch tensor
		pcs = torch.from_numpy(pcs).to(device)

		return {"mean": means, "std": stds, "pcs": pcs}


if __name__ == "__main__":
	# run a simple test

	# config
	model_name = ["bert-large-uncased_0_seeds7_max_rank"][-1]
	feature = ["complexity", "formality", "intensity", "figurativeness", "combined"][-2]

	predicted_class = PREDICTED_CLASS_MAP[feature]

	# load model
	config_frn = f"source/configuration/config_files/{feature}/{model_name}.json"
	config = Config.from_json_file(config_frn)

	if "freq" in model_name:
		featurizer = FreqFeaturizer(config)
	else:
		featurizer = LexicalFeaturizer(config)
		featurizer.load_LM()
		featurizer.generate_dvecs()

	texts = ["make the child proud",
	         "fill the child with pride"]
	for feature in featurizer.features:
		feature_scores = featurizer.predict_feature_batch(texts, feature)
		print(f"Feature scores for {feature}: {feature_scores}")

	examples = [{"0": texts[0], "1": texts[1]}]
	preds = featurizer.compare_feature_batch(examples, feature)
	for i, example in enumerate(examples):
		print(example["0"], example["1"], preds[i])