Wine Quality Analysis
This project involves analyzing a wine dataset to predict wine quality using machine learning techniques. The analysis includes data preprocessing, training machine learning models, and evaluating their performance.

Table of Contents
Project Description
Data Description
Installation
Usage
Results
Contributing
License
Project Description
The goal of this project is to classify the quality of wine based on various chemical properties. We utilize two machine learning algorithms: K-Nearest Neighbors (KNN) and Logistic Regression. The project includes data preprocessing steps such as renaming columns, combining classes, and splitting the data into training and testing sets.

Data Description
The dataset used in this project is a CSV file named wine.data with the following columns:

Quality
Alcohol
Malic Acid
Ash
Alcalinity of Ash
Magnesium
Total Phenols
Flavanoids
Nonflavanoid Phenols
Proanthocyanins
Color Intensity
Hue
OD280/OD315
Proline
The target variable is Quality, which has been converted to a binary outcome (0 or 1).

Installation
To run this project, you need to have R installed on your system along with the following R packages:

class
glmnet
ggplot2
pROC
You can install these packages using the following commands:

R code
install.packages("class")
install.packages("glmnet")
install.packages("ggplot2")
install.packages("pROC")

Usage
Load the dataset into the workspace:

R
Copy code
wine.data <- read.csv("~/Desktop/wine.data")
Run the R script wine_data.r to execute the entire analysis:

R
Copy code
source("wine_data.r")
The script performs the following tasks:

Loads and preprocesses the data.
Splits the data into training and testing sets.
Fits a K-Nearest Neighbors (KNN) model.
Fits a Logistic Regression model.
Evaluates the performance of both models.
Visualizes the data and the ROC curve.
Results
The performance metrics for the models include accuracy, precision, recall, and F1 score. The script outputs these metrics for both the KNN and Logistic Regression models.

Additionally, the script generates a box plot showing alcohol content by wine quality and an ROC curve for the logistic regression model.

Contributing
Contributions are welcome! If you have any suggestions or improvements, please submit a pull request or open an issue.

License
