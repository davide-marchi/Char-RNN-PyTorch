from CharRNN import train, validate
import itertools
import json

hidden_size_list = [64, 128]
num_layers_list = [2, 3]
dropout_rate_list = [0.2, 0.5]

results = []

for hidden_size, num_layers, dropout_rate in itertools.product(hidden_size_list, num_layers_list, dropout_rate_list):
    train(hidden_size, num_layers, dropout_rate)
    tr_loss = validate(hidden_size, num_layers, dropout_rate, './data/tr_reviews.txt')
    vl_loss = validate(hidden_size, num_layers, dropout_rate, './data/vl_reviews.txt')
    dict_results = {'hidden_size': hidden_size, 'num_layers': num_layers, 'dropout_rate': dropout_rate, 'tr_loss': tr_loss, 'vl_loss': vl_loss}
    results.append(dict_results)
    # Save results as JSON with indentation
    with open('./results.json', 'w') as file:
        json.dump(results, file, indent=4)
