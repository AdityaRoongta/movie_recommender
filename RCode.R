#### setting working directory ####
#---------------------------------------------------------
setwd("/Users/adityaroongta/Google Drive/Education/SP Jain/Week 19 - Data Mining and Recommender Systems/Project/ml-latest-small/")


#### installing and loading requisite packages ####
#---------------------------------------------------------
install.packages("recommenderlab")
library(recommenderlab)
library(data.table)
library(sqldf)


#### Reading and exploring ratings data ####
#---------------------------------------------------------
ratings <- read.csv("ratings.csv", header = T)
head(ratings)
length(unique(ratings$userId))  # This shows total unique users = 678
length(unique(ratings$movieId))  # This shows total unique movies = 9066


#### Dividing ratings into training and test data sets ####
#---------------------------------------------------------
## removing redundant last column
ratings <- ratings[,-c(4)]
head(ratings)

## Dividing the data into train and test sets using stratefied random sampling
set.seed(123) # setting the seed to make the partition reproducible
# culling out 80% (random stratified on userID) of the dataset as training dataset
ratings_train <- stratified(indt = ratings, group = "userId", size = 0.8)
# Remaining 20% of the data as test dataset
ratings_test <- sqldf("select * from ratings except select * from ratings_train")
nrow(ratings_train) # checking the split size
nrow(ratings_test) # checking the split size


#### Re-formating datasets into matrix form ####
#---------------------------------------------------------
g_train <- acast(ratings_train, userId ~ movieId)
head(g_train[,1:10])
class(g_train)
ncol(g_train)

g_test <- acast(ratings_test, userId ~ movieId)
head(g_test[,1:10])
class(g_test)
ncol(g_test)

## Convert the above matrices into realRatingMatrix data structure
## realRatingMatrix is a recommenderlab sparse-matrix
r_train <- as(g_train, "realRatingMatrix")
r_test <- as(g_test, "realRatingMatrix")


#### Creating a recommender object (model) ####
####   The three lines of code pertain to three different algorithms.
####    All the three algorithms use UBCF: User-based collaborative filtering
####     Parameter 'method', which decides similarity measure, differs for all three
####      Method 1 - Cosine, Method 2 - Jaccard and Method 3 - Pearson
#---------------------------------------------------------

## Creating model using 'Cosine' similarity measure
rec_cosine <- Recommender(r_train[1:nrow(r_train)],method="UBCF", param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=1))
## Creating model using 'Jaccard' similarity measure
rec_jaccard <- Recommender(r_train[1:nrow(r_train)],method="UBCF", param=list(normalize = "Z-score",method="Jaccard",nn=5, minRating=1))
## Creating model using 'Pearson Correlation' similarity measure
rec_pearson <- Recommender(r_train[1:nrow(r_train)],method="UBCF", param=list(normalize = "Z-score",method="Pearson",nn=5, minRating=1))


#### Creating Predictions ####
# This prediction does not predict movie ratings for test.
#   But it fills up the user 'X' item matrix so that
#    for any userid and movieid, one can find predicted rating
#     'type' parameter decides whether you want ratings or top-n items
#       Here, we choose ratings
#---------------------------------------------------------

pred_cosine <- predict(rec_cosine, r_train[1:nrow(r_train)], type="ratings")
pred_cosine

pred_jaccard <- predict(rec_jaccard, r_train[1:nrow(r_train)], type = "ratings")
pred_jaccard

pred_pearson <- predict(rec_pearson, r_train[1:nrow(r_train)], type="ratings")
pred_pearson

## Converting all our predictions into list, user-wise
pred_cosine_list <- as(pred_cosine, "list")
pred_jaccard_list <- as(pred_jaccard, "list")
pred_pearson_list <- as(pred_pearson, "list")


#### Create submission files from the above models ####
#     We will create three different submission files
#      One each for our three prediction models
#---------------------------------------------------------
## Creating an empty object for predicted ratings for test dataset (One each for three models)
predicted_ratings_cosine <- NULL
predicted_ratings_jaccard <- NULL
predicted_ratings_pearson <- NULL

## For all users in test file, one by one, extracting the predicted ratings
## The below function needs to be run thrice, once each for three models
for (u in 1:length(ratings_test[,1])){
    
    # Reading userid and movieid from columns 1 and 2 of test dataset
    userid <- ratings_test[u,1]
    movieid <- ratings_test[u,2]
    
    # Get as list & then convert to data frame all recommendations for user: userid
    # This is the first part which needs to be iterated thrice, once for each of the three models.
    u1 <- as.data.frame(pred_pearson_list[[userid]])
    
    # Creating a second column ('id') in the data-frame u1 and populate it with row-names
    # Since the rownames in u1 are by movie-ids
    # We use row.names() function to create this column
    u1$id <- row.names(u1)
    
    # Now accessing movie ratings in column 1 of u1
    x <- u1[u1$id == movieid,1]
    
    # If no ratings found, assigning 0.
    # This is the second part which needs to be iterated thrice, once for each of the three models.
    if (length(x)==0){
        predicted_ratings_pearson[u] <- 0
    }
    else{
        predicted_ratings_pearson[u] <- x
    }
}

## Column-binding predicted ratings to the test rating files
## Again, this will need to be iterated thrice, once each for the three models
tx <- cbind(ratings_test[,1:3],round(predicted_ratings_pearson))
# Writing to a csv file: PredictionFile.csv in local folder
write.table(tx,file="PredictionFile_pearson.csv",row.names=FALSE,col.names=FALSE,sep=',')


#### Checking the performance of our models using NMAE ####
#---------------------------------------------------------

## For Cosine

# Importing prediction file for cosine
CosineData <- read.csv("PredictionFile_cosine.csv", header = FALSE)
Cosine_Actual <- CosineData[,3]  # Extracting actual ratings
Cosine_Predicted <- CosineData[,4]  # Extracting predicted ratings

# Using the MAE function of recommenderlab package to calculate the MAE
mae_cosine <- MAE(true = Cosine_Actual, predicted = Cosine_Predicted)
mae_cosine  # Retuns a value of 0.8359996

# The Min and Max ratings are common for all three models as the testing set is same across all 3 models
max_rating <- max(Cosine_Actual)
min_rating <- min(Cosine_Actual)

# Calculating NMAE = MAE/(max rating - min rating)
NMAE_cosine <- mae_cosine/(max_rating - min_rating )
NMAE_cosine  # Retuns a value of 0.1857777


## For Jaccard

# Importing prediction file for jaccard
JaccardData <- read.csv("PredictionFile_jaccard.csv", header = FALSE)
Jaccard_Actual <- JaccardData[,3]  # Extracting actual ratings
Jaccard_Predicted <- JaccardData[,4]  # Extracting predicted ratings

# Using the MAE function of recommenderlab package to calculate the MAE
mae_jaccard <- MAE(true = Jaccard_Actual, predicted = Jaccard_Predicted)
mae_jaccard  # Retuns a value of 0.8362996

# Calculating NMAE = MAE/(max rating - min rating)
NMAE_jaccard <- mae_jaccard/(max_rating - min_rating )
NMAE_jaccard # Retuns a value of 0.1858443


## For Pearson

# Importing prediction file for pearson
PearsonData <- read.csv("PredictionFile_pearson.csv", header = FALSE)
Pearson_Actual <- PearsonData[,3]  # Extracting actual ratings
Pearson_Predicted <- PearsonData[,4]  # Extracting predicted ratings

# Using the MAE function of recommenderlab package to calculate the MAE
mae_pearson <- MAE(true = Pearson_Actual, predicted = Pearson_Predicted)
mae_pearson  # Retuns a value of 0.8360996

# Calculating NMAE = MAE/(max rating - min rating)
NMAE_pearson <- mae_pearson/(max_rating - min_rating )
NMAE_pearson # Retuns a value of 0.1857999
