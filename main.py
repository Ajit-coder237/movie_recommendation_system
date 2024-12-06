# Step 1: Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 2: Load the Dataset
movies = pd.read_csv('./netflix_titles.csv')

# Step 3: Data Preprocessing
# Drop rows with missing values in relevant columns
movies = movies.dropna(subset=['listed_in', 'description'])

# Combine 'listed_in' (genres) and 'description' into a single 'content' column
movies['content'] = (movies['listed_in']*2) + ' ' + movies['description']

# Step 4: Text Vectorization
# Initialize the TF-IDF Vectorizer (ignore common English words)
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

# Fit and transform the "content" column
tfidf_matrix = tfidf.fit_transform(movies['content'])

# Step 5: Compute Cosine Similarity
# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of movie titles to indices
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Step 6: Recommendation Function
def recommend_movies(title, cosine_sim=cosine_sim, num_recommendations=5):
    """
    Recommend movies based on content similarity.

    Parameters:
        title (str): The title of the movie to base recommendations on.
        cosine_sim (ndarray): Precomputed cosine similarity matrix.
        num_recommendations (int): Number of similar movies to recommend.

    Returns:
        list: Titles of the recommended movies.
    """
    # Get the index of the movie matching the title
    idx = movie_indices.get(title, None)
    if idx is None:
        return f"Movie '{title}' not found in the dataset."
    
    # Get similarity scores for the movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top similar movies
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices_list = [i[0] for i in sim_scores]
    
    # Return the titles of the recommended movies
    return movies['title'].iloc[movie_indices_list].tolist()

# Step 7: Test the Recommendation System
# Example: Recommend movies similar to "Breaking Bad"
movie_title = input("Enter the title of the movie") # Replace with a movie title from the dataset
recommendations = recommend_movies(movie_title)

# Print the recommendations
if isinstance(recommendations, list):
    print(f"Movies similar to '{movie_title}':")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print(recommendations)
