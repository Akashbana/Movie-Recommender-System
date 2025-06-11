# Problem Statement

Aim is to build:

1. a ***personalized movie recommender system*** that predicts user preferences using ***user ratings, demographics, and movie details***
2. The system should accurately suggest movies to users, enhancing ***user engagement***, increasing satisfaction, and creating a more intuitive movie-watching experience

# EDA

<img src="Pictures/info.png" alt="Data" width="400"/> 

* Some of the columns like user id, age, ratings, etc have been assigned ***'object'*** data type which is wrong. Therefore, will be changed to ***'int'***
* Timestamp is in ***unix*** form which will be converted to ***date_time***

1. Rows: ***10,00,209*** ratings
2. Users: ***6,040 (UserID from 1 to 6040)***
3. Movies: ***~3952 unique Movie_ids***
4. Ratings: Range from 1 to 5, with a ***mean of 3.58*** → generally positive ratings
5. Age: Mean age ~29.7; most users aged ***25–35***. Age 1 may need cleaning
6. Occupation: Numeric codes (0–20); further decoding needed
7. Timestamp: Ratings span from ***Apr 2000 to Feb 2003*** — nearly ***3 years*** of data

**Uni-variate Analysis:**

<img src="Pictures/eda1.png" alt="Data" width="1000"/>

<img src="Pictures/eda2.png" alt="Data" width="1000"/>

1. No. of Genres - 301
2. No. of unique locations ( zipcodes ) - 3439
3. Gender: ***72% Male***, 28% Female
4. Age: Majority (~35%) are aged ***25***
5. Occupation: Most users belong to occupations ***4, 0, and 7***, each around 11–13%
6. Rating: Around ***80%*** of users given rating ***>= 3***

# Data Pre-Processing

***Missing Values:***

<img src="Pictures/missing values.png" alt="Data" width="400"/> 

* No missing values in the data

***Extracting Time Based Features:***

      df['year'] = df['Timestamp'].dt.year
      df['month'] = df['Timestamp'].dt.month
      df['day'] = df['Timestamp'].dt.day
      df['day_of_week'] = df['Timestamp'].dt.day_name()

<img src="Pictures/data.png" alt="Data" width="1000"/> 

# Model Building

## Collaborative Filtering

* Collaborative Filtering is a recommendation technique that suggests items to a user based on the preferences of similar users (user-based) or similar items (item-based)
* It relies on user-item interaction data (like ratings or clicks) without requiring item content. This approach helps uncover hidden patterns and personalized recommendations even for diverse item types

***User-Item Matrix:***

Each row represents a user and each column represents an item. The values are known interactions (e.g., ratings), and the goal is to predict ***missing values***

      # user-item matrix
      
      user_item_matrix = df.pivot_table(index = 'UserID', columns = 'Movie_id', values = 'Rating') # user interactions
      user_item_matrix

***User-User Similarity Matrix:***

It is a square matrix where each cell (u,v) contains the similarity score between users u & v, based on their interactions (e.g., ratings or clicks) across items

<img src="Pictures/user user matrix formula.png" alt="Data" width="700"/> 

      # Cosine similarity - computes similarity between rows
      
      from sklearn.metrics.pairwise import cosine_similarity 
      
      user_similarity = cosine_similarity(user_item_matrix.fillna(0))  # user-user similarity
      user_similarity_df = pd.DataFrame(user_similarity, index = user_item_matrix.index, columns = user_item_matrix.index)
      user_similarity_df

<img src="Pictures/user user matrix.png" alt="Data" width="700"/> 

***User Based CF:***

User-Based Collaborative Filtering recommends items by finding users with similar behavior and using their ratings to predict unknown ones

<img src="Pictures/user based cf formula.png" alt="Data" width="700"/> 

      #Ratings
      predicted_ratings = user_item_matrix.copy() # Empty DataFrame to store predicted ratings
      # Loop over every user and every movie
      for user in user_item_matrix.index:
          for movie in user_item_matrix.columns:
              if pd.isna(user_item_matrix.loc[user, movie]):
                  rated_by_users = user_item_matrix[movie].dropna().index  # users who have rated this movie
      
                  sim_scores = user_similarity_df.loc[user, rated_by_users]  # All users considered who have rated 
                  ratings = user_item_matrix.loc[rated_by_users, movie]  # rating by each user
                  
                  numerator = np.dot(sim_scores, ratings)  # similarity btw users x ratings by those users
                  denominator = sim_scores.abs().sum()
                  
                  if denominator != 0:
                      predicted_ratings.loc[user, movie] = numerator / denominator
                  else:
                      predicted_ratings.loc[user, movie] = np.nan
      
          rated_movies = user_item_matrix.loc[user].dropna().index
          predicted_ratings.loc[i, rated_movies] = np.NaN
      
      predicted_ratings

1. CF is taking a lot of time to compute ratings ---> ***O(n2)***
2. Instead, select ***top K users*** for ratings computation (dot product) ---> ***O(k x n)***

***Item-Item Similarity Matrix:***

It is a square matrix where each cell (i,j) represents the similarity between items j, calculated using user ratings across those items

<img src="Pictures/item item matrix formula.png" alt="Data" width="700"/>

      #item-item similarity matrix
      item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))  # computes similarity between rows i.e. user ids
      item_similarity_df = pd.DataFrame(item_similarity, index = user_item_matrix.columns, columns = user_item_matrix.columns)
      item_similarity_df

***Item Based CF:***

* Item-Based Collaborative Filtering recommends items by finding similar items a user has already interacted with
* It assumes that if a user liked one item, they'll likely enjoy similar ones

<img src="Pictures/item based cf formula.png" alt="Data" width="700"/>

      
      # Ratings
      predicted_ratings_item_based = user_item_matrix.copy() # Empty DataFrame to store predicted ratings
      
      for user in user_item_matrix.index:
          movies_watched = user_item_matrix.loc[user].dropna().index
          for movie in user_item_matrix.columns:
              if movie not in movies_watched:
                  rated_movies = user_item_matrix.loc[user, movies_watched]
                  similarities = item_similarity_df.loc[movie, movies_watched]
                  ratings = np.dot(rated_movies, similarities) / np.sum(similarities)  # computing ratings for (user, movie)
                  predicted_ratings_item_based.loc[user, movie] =  ratings  # updating ratings in the matrix
      
          predicted_ratings_item_based.loc[user, movies_watched] = np.NaN
      
      predicted_ratings_item_based

## Matrix Factorization
