import matplotlib.pyplot as plt
import numpy as np
import json
from CharRNN import validate


# Load results
with open('./results.json', 'r') as file:
    results = json.load(file)

# Prepare data for plotting
labels = ['HS 64, NL 2, DR 0.2', 'HS 64, NL 3, DR 0.2', 'HS 128, NL 2, DR 0.2', 'HS 128, NL 3, DR 0.2',
          'HS 64, NL 2, DR 0.5', 'HS 64, NL 3, DR 0.5', 'HS 128, NL 2, DR 0.5', 'HS 128, NL 3, DR 0.5']
tr_loss = [r['tr_loss'] for r in results]
vl_loss = [r['vl_loss'] for r in results]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, tr_loss, width, label='Training Loss')
rects2 = ax.bar(x + width/2, vl_loss, width, label='Validation Loss')

ax.set_ylim([0, 1.7])  # Set the y-axis limits
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Loss')
ax.set_title('Loss by hyperparameters and dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

fig.tight_layout()

plt.savefig('results.png')

# Find the best model
best_model = min(results, key=lambda x: x['vl_loss'])

# Extract the hyperparameters of the best model
best_hidden_size = best_model['hidden_size']
best_num_layers = best_model['num_layers']
best_dropout_rate = best_model['dropout_rate']

print(f"Best hidden size: {best_hidden_size}")
print(f"Best num layers: {best_num_layers}")
print(f"Best dropout rate: {best_dropout_rate}")

# Validate the best model
ts_loss = validate(best_hidden_size, best_num_layers, best_dropout_rate, './data/ts_reviews.txt')
print(f'Test loss for the best model: {ts_loss}')
# OUT: Test loss for the best model: 1.2589578165031023