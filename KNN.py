import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load song data
df = pd.read_csv('SONGS.csv')

# Combine all your features into a single string
df['combined_features'] = df['SONGS'] + df['GENRES']  # Add more features if you have

# Create a TF-IDF vectorizer. This will convert your combined features into a matrix of TF-IDF features.
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Calculate the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Let's say you want to find similar songs for a specific song at index 1 in the dataframe
selected_song_index = 7

# Get the cosine similarities of the selected song with all other songs
similarities = cosine_sim_matrix[selected_song_index]

# Sort the similarities in descending order and get the indices of the most similar songs
most_similar_song_indices = similarities.argsort()[::-1]

# Exclude the selected song itself from the recommended songs
recommended_songs = most_similar_song_indices[1:]

# Print the recommended songs
print(df.loc[recommended_songs])
