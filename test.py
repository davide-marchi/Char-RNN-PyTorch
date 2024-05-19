import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from CharRNN import RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(temperature=1.0):
    ############ Hyperparameters ############
    hidden_size = 128   # size of hidden state
    num_layers = 3      # num of layers in LSTM layer stack
    dropout_rate = 0.2  # dropout rate
    op_seq_len = 1000   # total num of characters in output test sequence
    
    load_path = "./preTrained/CharRNN_reviews_" + str(hidden_size) + "_"+ str(num_layers) + "_" + str(dropout_rate) + ".pth"
    data_path = "./data/ts_reviews.txt"
    #########################################
    
    # load the text file
    data = open(data_path, 'r').read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print("----------------------------------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")
    
    # char to index and idex to char maps
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    # convert data from chars to indices
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]
    
    # data tensor on device
    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)
    
    # create and load model instance
    rnn = RNN(vocab_size, vocab_size, hidden_size, num_layers, dropout_rate).to(device)
    rnn.load_state_dict(torch.load(load_path))
    rnn.eval()
    print("Model loaded successfully !!")
    
    # initialize variables
    data_ptr = 0
    hidden_state = None
    
    # randomly select an initial string from the data
    rand_index = np.random.randint(data_size - 11)
    input_seq = data[rand_index : rand_index + 9]
    
    # compute last hidden state of the sequence 
    _, hidden_state = rnn(input_seq, hidden_state)
    
    # next element is the input to rnn
    input_seq = data[rand_index + 9 : rand_index + 10]
    
    print("Temperature: ", temperature)

    # generate remaining sequence
    print("----------------------------------------")
    while True:

        # Forward pass
        output, hidden_state = rnn(input_seq, hidden_state)
        
        # Construct categorical distribution and sample a character
        output = F.softmax(torch.squeeze(output) / temperature, dim=0)
        dist = Categorical(output)
        index = dist.sample().item()
        
        # Print the sampled character
        print(ix_to_char[index], end='')
        
        # next input is current output
        input_seq[0][0] = index
        data_ptr += 1
        
        if data_ptr  > op_seq_len:
            break
        
    print("\n----------------------------------------")
   
if __name__ == '__main__':
    test()
    test(temperature=0.1)
    test(temperature=0.5)
    test(temperature=0.6)
    test(temperature=10)