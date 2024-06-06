# Cab-Booking-Prediction
Python Machine Learning project using scikitlearn,pandas,numpy,matplotlib

# Problem Statement
Build a Machine Learning model to predict the total count of cabs booked in each hour by the new data

Cab booking system is the process where renting a cab is automated through an app throughout a city.Using this app people can book a cab from one location to another location.  
Being a cab booking app company, exploiting an understanding of cab supply and demand could increase the efficiency of their service and enhance user experience by minimizing waiting time.
#### Objective
To combine historical usage pattern along with the open data sources like weather data to forecast cab booking demand in a city.
#### Process Flow: 
You will be provided with hourly renting data span of two years. 
Data is randomly divided into train and test set. You must predict the total count of cabs booked in each hour covered by the testset

# Analysis Approach

## Data Pre-Processing
Contains the initial exploration of data like 
*   Load training and testing data into DataFrame
*   Inference about the data -info
*   Find Missing Values
*   Feature Engineering 
    * Create new columns hour,weekday,month
    * Outliers Analysis and removing outliers
*   Data Visualization 
    * Correlation Analysis
    * Univariate
      * Distribution of target variable
      * Distribution of continuous variables
    * BiVariate-Distribution of target variable vs categorical variables
    * Multivariate - Distribution of all continuous variables vs other continuous variables
*   Keep only 1 column for columns having high correlation
*   One-hot encoding/dummification of the categorical variables
*   Dealing cyclic features - hour,month,season,weekday
*   Generating Cramer’s V pairwise matrix plot using `association_metrics` library

## Modeling
* Split the data into training and testing using train_test_split
* Standardize the data by applying fit on training data and transform on train and test data
* Regression algorithms shown below are used to build the model 
  * Linear Regression
  * Support Vector Machines(Linear)
  * KNN
  * NaiveBaye's
  * Decision Trees
  * Random Forest
  * Bagging
  * XGBoost,AdaBoost,GradientBoost are used to build the model
  * MLPRegressor
* Model Evaluation Metrics
  * R2score
  * AdjustedR2score
  * RootMeanSquaredError are calculated
* Feature Importance from the model is also displayed
* Ensembling models like RandomForest and Boosting and MLPRegressor gave the best results among all the algorithms used
## Research On Cyclic Features
  * Transforming hour,month into cos and sin and encoding season,weekDay improved accuracy for all tree based algorithms
  * Encoding hour,month,season,weekDay improved accuracy for MLPRegressor,LinearRegression,NaiveBayes,knn and decreased for tree based algormiths
  * Transforming hour,month,season,weekDay into cos and sin improved accuracy for knn,SVR algorithms


