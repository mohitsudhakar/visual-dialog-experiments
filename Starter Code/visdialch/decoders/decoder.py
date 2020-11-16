import torch
from torch import nn
import numpy as np
# from gensim.test.utils import common_texts
# from gensim.models import Word2Vec
from visdialch.utils import DynamicRNN
from visdialch.utils.util import get_pretrained_weights

class DiscriminativeDecoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config 
        #self.word_embed = nn.Embedding(len(vocab),  config["word_embedding_size"], padding_idx=vocab.PAD_INDEX)
        weights = get_pretrained_weights(vocab)
        self.word_embed = nn.Embedding.from_pretrained(weights)
        self.option_rnn = nn.LSTM(config["word_embedding_size"], config["lstm_hidden_size"], config["lstm_num_layers"], 
                            batch_first=True, dropout=config["dropout"])

        self.option_rnn = DynamicRNN(self.option_rnn)

    def forward(self, encoder_output, batch):
        # batch_size x num_rounds x num_options x max_seq_length
        opt = batch["opt"]
        batch_size, num_rounds, num_options, max_seq_length = opt.shape
        # (batch_size*num_rounds*num_options) x max_seq_length
        opt = opt.reshape(-1, opt.shape[-1])
        # (batch_size*num_rounds*num_options) x max_seq_length x embedding_size
        opt_emb = self.word_embed(opt)

        # Running the Decoder rnn on opt_emb to remove max_deq_len dimension
        # (batch_size*num_rounds*num_options) x lstm_hidden_size
        _, (opt_emb, _) = self.option_rnn(opt_emb, batch["opt_len"])

        # Reshape the encoder output to match with options embedding
        # batch_size x num_rounds x num_options x lstm_hidden_size
        encoder_output = encoder_output.unsqueeze(2).repeat(1,1, num_options, 1)

        # Get the dot product opt_emb & encoder_output
        # (batch_size*num_rounds*num_options) x lstm_hidden_size
        encoder_output = encoder_output.reshape(-1, self.config["lstm_hidden_size"])
        #print("encoder_output:", encoder_output.shape)
        #print("opt_emb:", opt_emb.shape)
        output_scores = torch.sum(opt_emb * encoder_output, 1)
        output_scores = output_scores.reshape(batch_size, num_rounds, num_options)
        return output_scores
