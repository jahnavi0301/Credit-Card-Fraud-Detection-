# importing the dta 
fraud <- read.csv("creditcard.csv")
fraud = fraud[2:31]

#Balancing the imbalanced data
library(mlr)
library(ParamHelpers)
library(foreach)
library(doParallel)
library(iterators)
library(parallel)
library(unbalanced)
data = ubBalance(X = fraud[, -30], Y = as.factor(fraud[, 30]), type = "ubSMOTE", percOver = 100, percUnder = 200, verbose = TRUE)
balancedData = cbind(data$X,data$Y)

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(balancedData$`data$Y`, SplitRatio = .8)
training = subset(balancedData , split == TRUE)
test = subset(balancedData, split == FALSE)

# Applying PCA
# install.packages('caret')
library(lattice)
library(ggplot2)
library(caret)
pca = preProcess(x = training[1:29], method = 'pca', pcaComp = 2)
training = predict(pca, training)
training = training[c(2,3,1)]
test = predict(pca, test)
test = test[c(2,3,1)]

# Fitting Kernel SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = training$`data$Y` ~ .,
                 data = training,
                 type = 'C-classification',
                 kernel = 'radial')
# Predicting the Test set results
y_pred = predict(classifier, newdata = test[-3])

# Making the Confusion Matrix
cm = table(test[, 3], y_pred)
cm


# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Kernel SVM (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

