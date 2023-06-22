from sklearn.cluster import KMeans

movies = [
    {"title":"Movie 1","features":[9, 1]},  # lots of action, little sci-fi
    {"title":"Movie 2","features":[8, 2]},  # lots of action, some sci-fi
    {"title":"Movie 3","features":[3, 7]},  # some action, lots of sci-fi
    {"title":"Movie 4","features":[4, 6]},  # some action, lots of sci-fi
    {"title":"Movie 5","features":[1, 8]},  # little action, lots of sci-fi
    {"title":"Movie 6","features":[9, 9]},  # lots of action, lots of sci-fi
]

feature_vectors = [movie["features"] for movie in movies]
kmeans = KMeans(n_clusters=2).fit(feature_vectors)

liked_movie = "Movie 6"
liked_movie_features = next(movie["features"] for movie in movies if movie["title"] == liked_movie)
liked_movie_cluster = kmeans.predict([liked_movie_features])[0]

recommended_movies = [movie for movie in movies if movie["title"] != liked_movie and kmeans.predict([movie["features"]])[0] == liked_movie_cluster]

print("We recommend:")
for recommended_movie in recommended_movies:
    print(recommended_movie["title"])
