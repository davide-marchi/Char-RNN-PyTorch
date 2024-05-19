import pandas as pd
import re

def convert_and_save_to_txt(data, file_path):

    reviews = data.str.encode('UTF-8', 'ignore').str.decode('UTF-8')

    # Remove everything that's not a letter, a number, basic punctuation, or dollar symbol
    reviews = reviews.apply(lambda x: re.sub(r'[^a-zA-Z0-9$€£.,!? ]', '', x))

    with open(file_path, 'w') as file:
        for review in reviews:
            file.write(review + '\n')

def control_data(data_path):
    # load the text file
    data = open(data_path, 'r').read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print("----------------------------------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")

    print(chars)


# Read the CSV file with UTF-8 encoding
df = pd.read_csv('data/Airline_Reviews.csv', dtype=str)

# Shuffle the rows using seed 42
df_shuffled = df.sample(frac=1, random_state=42)

# Divide the shuffled rows into two series
num_rows = len(df_shuffled)
train_size = int(0.7 * num_rows)
validation_size = int(0.2 * num_rows)

train_data = df_shuffled[:train_size]
validation_data = df_shuffled[train_size:train_size+validation_size]
test_data = df_shuffled[train_size+validation_size:]

convert_and_save_to_txt(train_data['Review'], 'data/tr_reviews.txt')
convert_and_save_to_txt(validation_data['Review'], 'data/vl_reviews.txt')
convert_and_save_to_txt(test_data['Review'], 'data/ts_reviews.txt')
control_data('data/tr_reviews.txt')
control_data('data/vl_reviews.txt')
control_data('data/ts_reviews.txt')