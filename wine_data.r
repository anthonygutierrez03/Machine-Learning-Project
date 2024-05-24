# Load the wine data into workspace
wine.data <- read.csv("~/Desktop/wine.data")

# Check attributes
str(wine.data)
summary(wine.data)

# Check the unique values in the response variable
table(wine.data$Quality)

# Rename columns
colnames(wine.data) <- c('Quality', 'Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD280_OD315', 'Proline')

# Display the updated column names
colnames(wine.data)

# Combine classes 2 and 3 into a single class
wine.data$Quality <- ifelse(wine.data$Quality == 1, 1, 0)

# Check the unique values again
table(wine.data$Quality)


# Convert the response variable to a binary outcome (0 or 1)
wine.data$Quality <- ifelse(wine.data$Quality == 1, 1, 0)

# Check the unique values again
table(wine.data$Quality)

# Split the data into training and testing sets
set.seed(123)  # for reproducibility
sample_indices <- sample(1:nrow(wine.data), 0.7 * nrow(wine.data))
train_data <- wine.data[sample_indices, ]
test_data <- wine.data[-sample_indices, ]

# Load required libraries
library(class)

# Fit KNN model
knn_model <- knn(train = train_data[, -1], test = test_data[, -1], cl = train_data$Quality, k = 5)

# Check the accuracy of KNN model on the test set
accuracy_knn <- sum(knn_model == test_data$Quality) / length(test_data$Quality)
cat("Accuracy of KNN model:", accuracy_knn, "\n")

# Load required library for logistic regression
library(glmnet)

# Ensure that the response variable is a factor
train_data$Quality <- as.factor(train_data$Quality)

# Fit logistic regression model
logistic_model <- glm(Quality ~ ., data = train_data, family = "binomial", maxit = 1000)

# Predictions on the test set
logistic_predictions <- predict(logistic_model, newdata = test_data, type = "response")
logistic_predictions <- ifelse(logistic_predictions > 0.5, 1, 0)

# Confusion matrix for logistic regression
conf_matrix_logistic <- table(Actual = test_data$Quality, Predicted = logistic_predictions)
conf_matrix_logistic

# Calculate accuracy, precision, recall, and F1 score
accuracy_logistic <- sum(diag(conf_matrix_logistic)) / sum(conf_matrix_logistic)
precision_logistic <- conf_matrix_logistic[2, 2] / sum(conf_matrix_logistic[, 2])
recall_logistic <- conf_matrix_logistic[2, 2] / sum(conf_matrix_logistic[2, ])
f1_score_logistic <- 2 * (precision_logistic * recall_logistic) / (precision_logistic + recall_logistic)

# Print metrics
cat("Accuracy of Logistic Regression:", accuracy_logistic, "\n")
cat("Precision of Logistic Regression:", precision_logistic, "\n")
cat("Recall of Logistic Regression:", recall_logistic, "\n")
cat("F1 Score of Logistic Regression:", f1_score_logistic, "\n")

# Confusion matrix for KNN
conf_matrix_knn <- table(Actual = test_data$Quality, Predicted = knn_model)
conf_matrix_knn

# Calculate accuracy for KNN
accuracy_knn <- sum(diag(conf_matrix_knn)) / sum(conf_matrix_knn)

# Print accuracy for KNN
cat("Accuracy of K-Nearest Neighbors (KNN):", accuracy_knn, "\n")

# Load required libraries
library(ggplot2)
library(pROC)

# Visualization
ggplot(train_data, aes(x = Quality, y = Alcohol, fill = as.factor(Quality))) +
  geom_boxplot() +
  labs(title = "Alcohol Content by Wine Quality",
       x = "Wine Quality", y = "Alcohol Content",
       fill = "Quality") +
  theme_minimal()

# ROC Curve
logistic_probs <- predict(logistic_model, newdata = test_data, type = "response")
roc_curve <- roc(test_data$Quality, logistic_probs)

# Plot Curve
plot(roc_curve, main = "ROC Curve - Logistic Regression", col = "blue", lwd = 2)
lines(c(0, 1), c(0, 1), lty = 2, col = "grey")