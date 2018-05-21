## Recommender System built on MovieLens dataset

This work is on popular ‘MovieLens’ database, which is a typical movie ratings database having users’ ratings for multiple movies. The project required us to do the following: -
 - Implement a Recommender System to predict ratings of movies not rated by users
 - Use a collaborative filtering technique to build the system
 - Use multiple methods/models of similarity between users and provide a comparative analysis of each

# Data Used
‘ratings.csv’ dataset, which is a dataset from MovieLens project. The data can be accessed from the following link
https://grouplens.org/datasets/movielens/latest/
The following is a summary of the dataset: -
Total unique users - 678
Total unique movies - 9066
Rating Range - 0 to 5
Rating format - Multiples of 0.5 in the Rating Range

# Building the Recommender System
The following steps are taken to build the recommender system on the above dataset: -
 1. Removing redundant data
    - Since the timestamp data is not going to be used, the column is removed before proceeding
 2. Splitting dataset into train and test sets
    - The dataset is divided into train set and test set in a ratio of 80:20 respectively
    - The division is done using stratified random sampling method
    - This method allows proportionate division of samples for each userID into both train and test sets while still   maintaining the randomness within userIDs
 3. Building models and predictions on training data
    - On the training data, User Base Collaborative Filtering (UBCF) technique is used to generate predictions for ratings of movies not rated by the users
    - UBCF is technique which uses one of the multiple methods to identify ’n’ nearest neighbours to a particular user, and then uses the average ratings of those n neighbours as the predicted rating for that particular user
    - The methods to identify the nearest neighbours include Cosine similarity, Jaccard and Pearson correlation
 4. Applying the Predictions on test data and checking for accuracy
    - Once the predictions are done on training data, we extract those predictions (userID wise) in test data set, thus giving us the actual and predicted movie ratings for each userID x movieID combination in test data set
    - After getting the actual vs. predicted ratings in test data set, Normalise Mean Absolute Error is calculated
		- The above steps are performed for each of the similarity measures, i.e. Cosine, Jaccard and Pearson
 5. Results and Interpretation
    - All the three methods produce almost equal NMAE. Therefore, none of the models can be picked over any other. One can use any of the above models for generating the predictions. The average error in all three cases is around 0.8 rating points
