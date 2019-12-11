import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class bilstm_attn(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, embed_dim, bidirectional, dropout):
        super(bilstm_attn, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.loss = nn.NLLLoss()
        self.softmax = nn.LogSoftmax()

        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2

        self.label = nn.Linear(hidden_size * self.layer_size, output_size)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_sentences, batch_size=None):
        input = input_sentences.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(self.layer_size, len(input_sentences), self.hidden_size))
        c_0 = Variable(torch.zeros(self.layer_size, len(input_sentences), self.hidden_size))

        lstm_output, (h_t, c_t) = self.lstm(input, (h_0, c_0))
        # attn_output = self.attention_net(lstm_output)
        output = h_t[0, :, :] + h_t[1, :, :]
        logits = self.softmax(self.label(output))
        return logits

class RNN(nn.Module):
	def __init__(self, input_dim, h, num_layer): # Add relevant parameters
		super(RNN, self).__init__()
		self.softmax = nn.LogSoftmax()
		self.loss = nn.NLLLoss()
		self.h = h
		self.num_layer = num_layer
		self.activation = nn.ReLU()
		self.rnn = nn.RNN(input_dim, h, num_layers=num_layer, nonlinearity='tanh')
		self.fc = nn.Linear(h, 2)

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def forward(self, inputs): 
		h0 = torch.zeros(self.num_layer, 1, self.h).requires_grad_()
		rnn_out, h_n = self.rnn(inputs, h0)
		y = self.fc(h_n[-1, -1, :])
		predicted_vector = self.softmax(y) 
		return predicted_vector
