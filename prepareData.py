import pandas as pd

# Read the CSV file
df = pd.read_csv('data/Airline_Reviews.csv')

# Extract the 'Review' column
reviews = df['Review']

# Save the reviews as a text file
with open('data/reviews.txt', 'w') as file:
    for review in reviews:
        file.write(review + '\n')