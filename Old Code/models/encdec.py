from torch import nn

from models.decoder.decoder import Decoder
from models.encoder.pretrained_bert_vgg import Encoder


class EncoderDecoder(nn.Module):
    def __init__(self, model, vocab):
        super().__init__()
        encoder = Encoder()
        decoder = Decoder()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        encoder_output = self.encoder(batch)
        decoder_output = self.decoder(encoder_output, batch)
        return decoder_output
