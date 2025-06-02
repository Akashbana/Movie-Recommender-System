# Problem Statement

Aim is to build a ***personalized movie recommender system*** that predicts user preferences using ***user ratings, demographics, and movie details.*** The system should accurately suggest movies to users, enhancing ***user engagement***, increasing satisfaction, and creating a more intuitive movie-watching experience

# EDA



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

1. Gender: ***72% Male***, 28% Female
2. Age: Majority (~35%) are aged ***25***
3. Occupation: Most users belong to occupations ***4, 0, and 7***, each around 11–13%
4. Rating: Around ***80%*** of users given rating ***>= 3***

# Data Pre-Processing

1. No missing values in the data
2. Time based features are extracted

