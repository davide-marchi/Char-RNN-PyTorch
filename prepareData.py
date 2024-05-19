import pandas as pd
import re

# Read the CSV file with UTF-8 encoding
df = pd.read_csv('data/Airline_Reviews.csv', dtype=str)
reviews = df['Review'].str.encode('UTF-8', 'ignore').str.decode('UTF-8')

# Remove everything that's not a letter, a number, basic punctuation, or dollar symbol
reviews = reviews.apply(lambda x: re.sub(r'[^a-zA-Z0-9$€£.,!? ]', '', x))

# Save the reviews as a text file
with open('data/reviews.txt', 'w') as file:
    for review in reviews:
        file.write(review + '\n')

data_path = "./data/reviews.txt"
# load the text file
data = open(data_path, 'r').read()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print("----------------------------------------")
print("Data has {} characters, {} unique".format(data_size, vocab_size))
print("----------------------------------------")

print(chars)