# Recommendation system

<img src = "data/banking.avif" width = 500  style="display: block; margin: 0 auto"/>

The aim of this project is to work on a recommendation system.

We are going to apply statistical and machine learning methods to construct our own recommenders and evaluate which recommender performe the best.

This project is separated in 2 different datasets:

1. First, we are going through a restaurants dataset and we are going to use simple analysis to see whether we can recommend to an user a restaurant based on the recommendations that person gave to a similar restaurant using the cuisine feature.

    In order to do that we are using the Pearson coefficient, so high coefficients means that restaurants are correlated. If restaurants are correlated, then the user is quite likely to like the restaurant suggested and give another good review.

    $$r = \frac{{}\sum_{i=1}^{n} (x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \overline{x})^2  \sum_{i=1}^{n}(y_i - \overline{y})^2}}$$

2. In this dataset, we are using Machine Learning. The choosen method is a Logistic Regression Recommender but we could also use others like Naive Bayes. We will see if a customer is good or bad based on its transaction history, user attribute data... so we can determine if we can grant a credit, personal loans or any other financial active.

    Logistic Regression estimates the probability of an event ocurring, based on of independent or predictor variables. For 5 variables:
    $$P(Y_i=1|X_i) = {\frac{exp(\beta_0 + \beta_1X_i + \beta_2X_2 + \beta_3X_3 + \beta_4X_4 + \beta_5X_5)}{1 + exp (\beta_0 + \beta_1X_i + \beta_2X_2 + \beta_3X_3 + \beta_4X_4 + \beta_5X_5)}}$$

## Motivation

Machine Learning has become one innovative and important tool when we are building a recomendation system. We can learn from a customer behavior in order to offer the best service. We are always using ML whithout even knowing. Leading tech companies are already using it to offer deals to customers.

Examples of Recommendation Engines:

* Product recommentation: Amazon
* Movie recommentation: Netflix
* Music recommentation: Spotify
* Social recommentation: Facebook

## Data Sources

Datasets are extracted from the [UCI Machine Learning repository](https://archive.ics.uci.edu/)

[1. Restaurants](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data>)

[2. Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)


> ### **Notes**
>
> * Restaurant dataset was obtained from a recommender system prototype. The task was to generate a top-n list of restaurants according to the consumer preferences.
> * Original Bank dataset `bank_full.csv` has been transformed beforehand to `bank_full_w_dummy_vars.csv`. Categorical text variables to binary categorical dummy variables. This can be done using for example the One Hot Encoding method from sklearn or the built-in method `get_dummy` from pandas library.
