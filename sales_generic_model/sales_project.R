rm(list = ls())
require(rpart) 
require(e1071) 
require(class)  
require(dplyr)
library(pROC)
library(ROCR)
require(randomForest)
library(lintr)
require(fst)


origin_df <- read.fst("D:/data/SalesKaggle3.fst") 


sum(is.na(origin_df))
df <- origin_df [is.na(origin_df$SoldFlag) == FALSE, ]    


# in order to enable me to create  balanced obs.  dataset between the sold and unsold
# 12996 obs. are originaly sold , 63000 obs are not sold and I used it to get 4 different samples of datasets in order to find the best one !
df <- arrange(df, desc(SoldCount))


# creating 4 options of balanced data set.
# for project using only the two best performances 
df1 <- read.fst("D:/data/sample1.fst")
df2 <- read.fst("D:/data/sample2.fst")

# feature selection to make it generic model
df1 <- select(df1, -File_Type, -Order, -New_Release_Flag, -SoldCount, -ItemCount)
df2 <- select(df2, -File_Type, -Order, -New_Release_Flag, -SoldCount, -ItemCount)


# Formula
# the product will be sold or not..
fmla <- as.formula("SoldFlag~.")

# turning into numeric
Nomeric_df <- function(df){
 numeric_features <- sapply(df, is.numeric)
 df[, numeric_features] <- scale(df[, numeric_features])
 df$SoldFlag <- ifelse(df$SoldFlag == "Yes", 1, 0)
 df$MarketingType <- ifelse(df$MarketingType == "D", 1, 0)

# Encoding the target feature as factor
 df$SoldFlag <- factor(df$SoldFlag, levels = c(0, 1))
 return(df)

}
df1 <- Nomeric_df(df1)
df2 <- Nomeric_df(df2)


# train & test :
# Generic machine learning code for the same train & test  for several models
train_test <- function(df){
  train_indx <- sample( 1:nrow (df), floor ((0.75) * nrow (df)))
  train_set <- df[train_indx, ]
  return (train_set)
}  
train_set_df1 <- train_test(df1)
train_set_df2 <- train_test(df2)


# Random Forest 

Random_Forest  <- function(fmla, train_set, test_set){
 rf_model <- randomForest(fmla, train_set)
 rf_model$importance 
 rf_predict <- predict(rf_model, test_set)
 
 # Making the Confusion Matrix, accuracy, recall, precision , roc
 confusion_matrix_rf <- table(test_set$SoldFlag, rf_predict)
 print("Random Forest cm : " )
 print(confusion_matrix_rf)
 accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
 print(paste0("Random Forest accuracy : ", accuracy_rf))
 recall.rf <- confusion_matrix_rf[2, 2] / (confusion_matrix_rf[2, 1] + confusion_matrix_rf[2, 2]) 
 print(paste0("Random Forest recall : ", recall.rf))
 precision.rf <- confusion_matrix_rf[2, 2] / (confusion_matrix_rf[1, 2] + confusion_matrix_rf[2, 2])   
 print(paste0("Random Forest precision : ", precision.rf))
 roc_rf <- roc(test_set$SoldFlag, order( rf_predict))
 pred <- prediction(order( rf_predict), test_set$SoldFlag)
 perf <- performance( pred, "tpr", "fpr")
 plot(perf, type = "l")
 print("Random Forest roc : ")
 print(roc_rf)

}


# SVM

SVM  <- function(fmla, train_set, test_set){
 svm_r <- svm(fmla, data=train_set,
                type="C-classification",
                kernel="radial")
 pred_r <- predict(svm_r, test_set)
 
 # Making the Confusion Matrix, accuracy, recall, precision , roc
 confusion_matrix_SVM <- table(test_set$SoldFlag, pred_r)
 print("svm  cm : ")
 print(confusion_matrix_SVM)
 accuracy_svm_r <- sum(diag(confusion_matrix_SVM)) / sum(confusion_matrix_SVM)
 print(paste0("svm accuracy : ", accuracy_svm_r))
 recall_svm_r <- confusion_matrix_SVM[2, 2] / (confusion_matrix_SVM[2, 1] + confusion_matrix_SVM[2, 2])
 print(paste0("svm recall : ", recall_svm_r))
 precision_svm_r <- confusion_matrix_SVM[2, 2] / (confusion_matrix_SVM[1, 2] + confusion_matrix_SVM[2, 2]) 
 print(paste0("svm precision : ", precision_svm_r))
 roc_svm_r <- roc(test_set$SoldFlag, order( pred_r))
 pred <- prediction(order( pred_r), test_set$SoldFlag)
 perf <- performance( pred, "tpr", "fpr")
 plot(perf, type = "l")
 print("SVM roc : ")
 print(roc_svm_r)
}

# Simple Decision Tree

Decision_Tree  <- function(fmla, train_set, test_set){
 dt_model_train <- rpart(fmla, train_set)
 dt_predict <- predict(dt_model_train, test_set, type = "class")

 # Making the Confusion Matrix, accuracy, recall, precision , roc
 confusion_matrix_dt <- table(test_set$SoldFlag, dt_predict)
 print("Decision Tree  cm : ")
 print( confusion_matrix_dt)
 accuracy_dt <- sum(diag(confusion_matrix_dt)) / sum(confusion_matrix_dt)
 print(paste0("Decision Tree accuracy : ", accuracy_dt))
 recall_dt <- confusion_matrix_dt[2, 2] / (confusion_matrix_dt[2, 1] + confusion_matrix_dt[2, 2])
 print(paste0("Decision Tree recall : ", recall_dt))
 precision_dt <- confusion_matrix_dt[2, 2] / (confusion_matrix_dt[1, 2] + confusion_matrix_dt[2, 2])
 print(paste0("Decision Tree precision : ", precision_dt))
 roc_dt <- roc(test_set$SoldFlag, order( dt_predict))
 pred <- prediction(order( dt_predict), test_set$SoldFlag)
 perf <- performance( pred, "tpr", "fpr")
 plot(perf, type = "l")
 print("Decision Tree roc : ")
 print(roc_dt)
 plot(dt_model_train)
 text(dt_model_train)

}



# logistic regression

Logistic_regression <- function(fmla, train_set, test_set){


 classifier <- glm(formula = fmla,
                  family = binomial,
                  data = train_set)


# Predicting the Test set results

 prob_pred <- predict(classifier, type = 'response', newdata =  test_set[-2])
 y_pred <- ifelse(prob_pred > 0.5, 1, 0)



# Making the Confusion Matrix, accuracy, recall, precision , roc

 cm <- table(test_set[, 2], y_pred > 0.5)
 print("Logistic Regression  cm : ")
 print( cm)
 accuracy_lr <- sum(diag(cm)) / sum(cm)
 print(paste0("Logistic Regression accuracy : ", accuracy_lr))
 recall_lr <- cm[2, 2] / (cm[2, 1] + cm[2, 2])
 print(paste0("Logistic Regression recall : ", recall_lr))
 precision_lr <- cm[2, 2] / (cm[1, 2] + cm[2, 2])
 print(paste0("Logistic Regression precision : ", precision_lr))
 roc_lr <- roc(test_set$SoldFlag, order( y_pred))
 pred <- prediction(order( y_pred), test_set$SoldFlag)
 perf <- performance( pred, "tpr", "fpr")
 plot(perf, type = "l")
 print("Logistic Regressione roc : ")
 print(roc_lr)


# Making the Confusion Matrix, accuracy, recall, precision , roc

}
while (TRUE){
 my.modoule <- function(modoule){
  modoule <- readline("choose your modoule 1. Logistic Regression 2. Random Forest 3. SVM  4. Decision Tree 5.Exit =" )
  if (modoule == 1){
    print("sample 1 results :")
    Logistic_regression(fmla, train_set_df1, setdiff(df1, train_set_df1))
    print("sample 2 results :")
    Logistic_regression(fmla, train_set_df2, setdiff(df2, train_set_df2))
  } else if (modoule == 2){
    print("sample 1 results :")
    Random_Forest(fmla, train_set_df1, setdiff(df1, train_set_df1))
    print("sample 2 results :")
    Random_Forest(fmla, train_set_df2, setdiff(df2, train_set_df2))
  } else if (modoule == 3){
    print("sample 1 results :")
    SVM(fmla, train_set_df1, setdiff(df1, train_set_df1))
    print("sample 2 results :")
    SVM(fmla, train_set_df2, setdiff(df2, train_set_df2))
  } else if (modoule == 4){
    print("sample 1 results :")
    Decision_Tree(fmla, train_set_df1, setdiff(df1, train_set_df1))
    print("sample 2 results :")
    Decision_Tree(fmla, train_set_df2, setdiff(df2, train_set_df2))
  } else if (modoule == 5){
    
    stop('\r      Thank you and Goodbye ' )
  }
 }
my.modoule(modoule)  
}



