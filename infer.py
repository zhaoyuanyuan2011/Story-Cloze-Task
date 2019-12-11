import os
import torch
import model
import numpy as np
from data_loader import fetch_test
from main import convert_to_vector_test

test_path = './test.csv'
out_path = './output.csv'
model_path = 'model/rnn_best.pth'
embed_dim = 300
hidden_size = 64
num_layer = 6

test_data = fetch_test(test_path)
test_data = convert_to_vector_test(test_data)
# test_id = get_id(test_path)
rnn = model.RNN(input_dim=embed_dim, h=hidden_size, num_layer=num_layer)
checkpoint = torch.load(model_path)
rnn.load_state_dict(checkpoint)

with open(out_path, "w+") as outfile:
    for temp_input in test_data:
        temp_input = torch.from_numpy(np.asarray([np.asarray(word) for word in temp_input])).unsqueeze(1)
        predicted = rnn(temp_input)
        predicted_label = int(torch.argmax(predicted))+1
        outfile.write(str(predicted_label))
        outfile.write('\n')