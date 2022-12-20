# 1. Description

This directory contains the implementation of a recommendation system for
the bookcrossing-dataset. It contains classes for loading the data, different
types of predictors and a recommender which uses a predictor to recommend 
movies to a specific user.

Data courtesy of Kaggle at [https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)

# 2. Structure

- The `UserItemData.py` file contains the UserItemData class,
  which is used for loading the ratings data.
- The `MovieData.py` file contains the MovieData class, which
  is used for mapping movie IDs to titles.
- The `Recommender.py` file contains the Recommender class, which
  recommends movies based on a given predictor.
- The `RandomPredictor.py` file contains the RandomPredictor class,
  which generates random values for predictions.
- The `AveragePredictor.py` file contains the AveragePredictor class
  which predicts movies based on average ratings.
- The `ViewsPredictor.py` file contains the ViewsPredictor class, which
  predicts movies based on the number of views.
- The `ItemBasedPredictor.py` file contains the ItemBasedPredictor class,
  which predicts movies based on similarities calculated by the adjusted
  cosine distance.
- The `SlopeOnePredictor.py` file contains the SlopeOnePredictor class,
  which predicts movies based on the Slope One method.
- The `MatrixFactorizationPredictor.py` file contains the MatrixFactorizationPredictor
  which uses the matrix factorization technique to predict movies. It also 
  visualises the results and matrix decomposition.

# 2.1 Optional tasks:

- The `STDPredictor.py` file contains the STDPredictor.py class, which
  predicts controversial movies based on the standard deviation.
- The `MyPredictions.py` file contains an implementation of predictions
  based on my predictions of specific movies.
- The `HybridPredictor.py` file contains the HybridPredictor class,
  which predicts movies by combining multiple predictors and averaging
  the values.

# 3. Commentary

Some predictions can definitely improved, for example, the matrix 
factorization predictor (spec. the rank) could be fine tuned to 
provide better predictions.