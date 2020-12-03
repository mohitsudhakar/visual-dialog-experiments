import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertTokenizer, BertModel

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[6] = torch.nn.Identity()

        self.bert_q = BertModel.from_pretrained('bert-base-uncased')
        self.bert_H = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 768)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, img, q, H):
        """
        Forward pass
        :param img: Image pixels transformed
        :param q: Question tokens
        :param H: Dialog history concat tokens
        :return: Output of Late Fusion encoder with vgg and bert
        """

        imgout = self.vgg(img)
        qout = self.bert_q(q)
        Hout = self.bert_H(H)
        out = torch.cat([imgout, qout, Hout], dim=1)
        return self.linear(out)

