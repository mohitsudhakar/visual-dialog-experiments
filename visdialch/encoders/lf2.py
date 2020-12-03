import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from visdialch.utils import DynamicRNN

class LateFusionEncoder(nn.Module):

	def __init__(self, config, vocabulary):

		super().__init__()
		self.config = config
		self.dropout = nn.Dropout(p=config["dropout"])

		self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )


		self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )

		self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )

		self.ques_rnn = DynamicRNN(self.ques_rnn)
		self.hist_rnn = DynamicRNN(self.hist_rnn)

		self.image_features_projection = nn.Linear(config["img_feature_size"], config["lstm_hidden_size"])
		self.attention_proj = nn.Linear(config["lstm_hidden_size"], 1)

		fusion_size = (
            config["img_feature_size"] + config["lstm_hidden_size"] * 2
        )
		self.fusion = nn.Linear(fusion_size, config["lstm_hidden_size"])

		nn.init.kaiming_uniform_(self.image_features_projection.weight)
		nn.init.constant_(self.image_features_projection.bias, 0)
		nn.init.kaiming_uniform_(self.fusion.weight)
		nn.init.constant_(self.fusion.bias, 0)

	def forward(self, batch):

		# Encoding Image
		# batch_size x num_proposal x img_feature
		img_features = batch["img_feat"]
		num_rounds = 10
		batch_size = img_features.shape[0]
		# print("Img feat: ", img_features.shape)

		# (bs x num_round) x (num_proposal) x (img_feature_shape => cnn feature shape)
		img_encoded_for_dot_product = img_features.unsqueeze(1).repeat(1, num_rounds, 1, 1).view((img_features.shape[0] * num_rounds), (img_features.shape[1]), (img_features.shape[-1]))
		# print(" img_encoded_for_dot_product:", img_encoded_for_dot_product.shape)
		# (bs x num_round) x (num_proposal) x (lstm_hidden_size)
		img_encoded = self.image_features_projection(img_encoded_for_dot_product)
		# print("img_encoded: ", img_encoded.shape)

		# Encoding Question
		# BS x num_round x max_seq_len
		question_features = batch["ques"]

		# Combining BS and num_round
		question_features = question_features.view((question_features.shape[0]*question_features.shape[1]), question_features.shape[2])
		# print("Combining BS and round for question: ", question_features.shape)

		# Using embedding to encode the words first
		# (BS x num_round) x max_seq_len x embedd_dim
		question_features = self.word_embed(question_features)
		print("Type of ques: ", type(question_features))
		# print("Using word embedding: ", question_features.shape)

		# # (BS x num_round) x max_seq_len x embedd_dim   => Already done
		# question_features = question_features.view((BS * num_round), max_seq_len, embedd_dim)
		
		# (BS x num_round) x lstm_hidden_size
		_, (question_features, _) = self.ques_rnn(question_features, batch["ques_len"])


		# For doing dot product we need question features to be similar in shape to img_features
		# (BS x num_round) x (num_proposal) x (lstm_hidden_size)
		question_features_for_dot = question_features.unsqueeze(1).repeat(1, img_features.shape[1], 1)
		# print("question_features_for_dot: ", question_features_for_dot.shape)

		# Encoding history
		# BS x num_round x (max_seq_len * 20)
		hist_features = batch["hist"]

		# Combining BS and num_round
		# (BS x num_round) x (max_seq_len * 20)
		hist_features = hist_features.view((hist_features.shape[0]*hist_features.shape[1]), hist_features.shape[2])
		# print("Combining BS and round for history: ", hist_features.shape)

		# Using embedding to encode the words first
		# (BS x num_round) x (max_seq_len * 20) x embedd_dim
		hist_features = self.word_embed(hist_features)
		# print("Hist features: ", hist_features.shape)

		# # (BS x num_round) x (max_seq_len*20) x embedd_dim
		# hist_features = hist_features.view((BS * num_round), max_seq_len*20, embedd_dim)
		
		# (BS x num_round) x lstm_hidden_size
		_, (hist_features, _) = self.hist_rnn(hist_features, batch["hist_len"])



		#  Doing dot product of question and image 
		#  (BS x num round) x (num_prop) x (lstm_hidden_size)
		product_question_image = (question_features_for_dot * img_encoded)
		product_question_image = self.dropout(product_question_image)
		# print("product_question_image: ", product_question_image.shape)

		# Finding weights
		# (BS x num round) x (num_prop)
		attention_weights = self.attention_proj(product_question_image).squeeze()
		attention_weights =  F.softmax(attention_weights, dim=-1)
		# print("attention_weights: ", attention_weights.shape)

		# For doing dot product with image the attention weight size should be similar to image size
		# (BS x num round) x (num_prop) x (img_feature_size)
		attention_weights = attention_weights.unsqueeze(-1).repeat(1,1,img_features.shape[-1])      #repeat(config["image_feature_size"])
		# print("Repeating attention_weights: ", attention_weights.shape)

		#  Doing dot product with image to find which proposal has the attention
		# shape: (BS x num round) x (img_feature_size)
		# attended_features = np.tensordot(img_features, attention_weights, 1) # Along axis = 1 i.e. number of proposal axis   CHECK
		attended_features = (attention_weights * img_encoded_for_dot_product).sum(1)
		# print("Final attended features: ", attended_features.shape)

		#  Doing concatenation:
		fused_vector = torch.cat((attended_features, question_features, hist_features), 1)
		fused_vector = self.dropout(fused_vector)

		# DOing normalization
		fused_embedding = torch.tanh(self.fusion(fused_vector))

		# Last mein BS and num_round ko tod dia
		fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)

		return fused_embedding

