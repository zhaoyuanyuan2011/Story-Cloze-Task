import os
import csv
import time
import torch
import model
import random
import sister
import numpy as np
from tqdm import tqdm
from torchnlp.word_to_vector import GloVe
from data_loader import fetch_data, fetch_data2

lr = 0.0001
epochs = 15
seed = 1111
train_path = 'train.csv'
dev_path = 'dev.csv'
model_path = 'model/rnn.pth'
dropout = 0
embed_dim = 300
hidden_size = 64
num_layer = 6
batch_size = 16
output_size = 2
bidirectional = True
weight_decay = 0.001

torch.manual_seed(seed)
def convert_to_vector_representation(data):
	glv = GloVe()
	vectorized_data = []
	for document, y in data:
		vector = []
		for word in document:
			word_embed = glv[word]
			vector.append(word_embed)
		vectorized_data.append((vector, y))
	return vectorized_data

def convert_to_vector_representation2(data):
	embedder = sister.MeanEmbedding(lang="en")
	vectorized_data = []
	for sentences, y in data:
		new_sent = []
		for sentence in sentences:
			new_sent.append(embedder(sentence))
		vectorized_data.append((new_sent, y))
	return vectorized_data

def convert_to_vector_test(data):
	embedder = sister.MeanEmbedding(lang="en")
	vectorized_data = []
	for sentences in data:
		new_sent = []
		for sentence in sentences:
			new_sent.append(embedder(sentence))
		vectorized_data.append(new_sent)
	return vectorized_data

def main():
	print("Fetching data")
	train_data, valid_data = fetch_data2(train_path, start=4), fetch_data2(dev_path, start=4)
	# print(train_data[1362])
	# print(train_data[1337])
	train_data = convert_to_vector_representation2(train_data)
	valid_data = convert_to_vector_representation2(valid_data)
	print("Vectorized data")

	lstm_attn = model.bilstm_attn(batch_size=batch_size,
								  output_size=output_size,
                                  hidden_size=hidden_size,
                                  embed_dim=embed_dim,
                                  bidirectional=bidirectional,
                                  dropout=dropout)
	rnn = model.RNN(input_dim=embed_dim, 
					h=hidden_size,
					num_layer=num_layer)
	# ffnn = model.FFNN(input_dim=embed_dim, h=hidden_size)
	# optimizer = torch.optim.Adam(lstm_attn.parameters(), lr=lr, weight_decay=weight_decay)
	optimizer = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=weight_decay)
	train_acc = valid_acc = 1
	epoch = 0

	while epoch < epochs:
		epoch += 1
		optimizer.zero_grad()
		loss = None
		correct = 0
		total = 0
		start_time = time.time()
		print("Training started for epoch {}".format(epoch))
		random.shuffle(train_data) # Good practice to shuffle order of training data
		N = len(train_data) 
		for minibatch_index in tqdm(range(N // batch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(batch_size):
				input_vector, gold_label = train_data[minibatch_index * batch_size + example_index]
				input_vector = torch.from_numpy(np.asarray([np.asarray(word) for word in input_vector])).unsqueeze(1)
				# predicted_vector = lstm_attn(input_vector)
				predicted_vector = rnn(input_vector)
				predicted_label = torch.argmax(predicted_vector)
				#if predicted_label != gold_label and epoch == 13:
				#	print(minibatch_index * batch_size + example_index)
				correct += int(predicted_label == gold_label)
				total += 1
				example_loss = rnn.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
				if loss is None:
					loss = example_loss
				else:
					loss += example_loss
			loss = loss / batch_size
			loss.backward()
			optimizer.step()
		print("Training completed for epoch {}".format(epoch))
		print("Training accuracy for epoch {}: {}".format(epoch, correct / total))
		print("Training time for this epoch: {}".format(time.time() - start_time))

		correct = 0
		total = 0
		start_time = time.time()
		print("Validation started for epoch {}".format(epoch))
		random.shuffle(valid_data) # Good practice to shuffle order of validation data
		N = len(valid_data) 
		for minibatch_index in tqdm(range(N // batch_size)):
			for example_index in range(batch_size):
				input_vector, gold_label = valid_data[minibatch_index * batch_size + example_index]
				input_vector = torch.from_numpy(np.asarray([np.asarray(word) for word in input_vector])).unsqueeze(1)
				# predicted_vector = lstm_attn(input_vector)
				predicted_vector = rnn(input_vector)
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
		print("Validation completed for epoch {}".format(epoch))
		print("Validation accuracy for epoch {}: {}".format(epoch, correct / total))
		print("Validation time for this epoch: {}".format(time.time() - start_time))
	torch.save(rnn.state_dict(), model_path)

if __name__ == '__main__':
	main()
