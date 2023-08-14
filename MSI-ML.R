#Tools - R
#K-fold cross validation (QDA, LDA, SVM, RF)
library(tidyverse)
library(caret)
library(cowplot)
library(MASS)
library(pROC)
FishData <- read.csv("path",header = T) #File path
for (i in c(1)) {FishData[,i] <- factor(FishData[,i])}
set.seed(20)
trains <- createDataPartition(y = FishData$sp,p = 0.70,list = F)
traindata <-FishData[trains,]
testdata <- FishData[-trains,]
 
form_clsm <-as.formula(
  paste0("sp ~ ", paste(colnames(traindata)[2:length(FishData)], collapse =" + "))
)
#cross validation
set.seed(34)
folds <- createFolds(y=traindata$sp, k=10)
max=0 
mum=0 

# QDA
for(i in 1:10){
  set.seed(34)
  fold_test <- traindata[folds[[i]],] 
  fold_train <- traindata[-folds[[i]],] 
  fold_pre <- qda(form_clsm, data = fold_train, method = "t") 
  fold_predict <- predict(fold_pre,type='response',newdata=fold_test) 
  fold_acc <- multiClassSummary(data.frame(obs = fold_test$sp,pred = fold_predict$class),lev = levels(fold_test$sp) )
  fold_auc <- multiclass.roc(response = fold_test$sp,predictor = fold_predict$posterior)
  cat("K:",i,'acc:',fold_acc[1],'auc:',fold_auc$auc,file = "file path",append = TRUE,fill = TRUE) #Save file path
}

# LDA
for(i in 1:10){
  fold_test <- traindata[folds[[i]],] 
  fold_train <- traindata[-folds[[i]],] 
  fold_pre <- lda(form_clsm, data = fold_train, method = "t")
  fold_predict <- predict(fold_pre,type='response',newdata=fold_test) 
  fold_acc <- multiClassSummary(data.frame(obs = fold_test$sp, pred = fold_predict$class),lev = levels(fold_test$sp) )
  fold_auc <- multiclass.roc(response = fold_test$sp,predictor = fold_predict$posterior)
  cat("K:",i,'acc:',fold_acc[1],'auc:',fold_auc$auc,file = "file path",append = TRUE,fill = TRUE)
}

# RF
library(randomForest)
for(i in 1:10){
  fold_test <- traindata[folds[[i]],] 
  fold_train <- traindata[-folds[[i]],] 
  fold_pre <- fit_rf_clsm <- randomForest(form_clsm, data = fold_train, ntree = 100, mtry = 4, importance = T,cost =2,gamma =0.025)
  fold_predict <- predict(fold_pre,type='response',newdata=fold_test)
  fold_predict_p <- predict(fold_pre,type='prob',newdata=fold_test)
  fold_acc <- multiClassSummary(data.frame(obs = fold_test$sp,pred = fold_predict),lev = levels(fold_test$sp) )
  fold_auc <- multiclass.roc(response = fold_test$sp,predictor = fold_predict_p)
  cat("K:",i,'acc:',fold_acc[1],'auc:',fold_auc$auc,file = "file path",append = TRUE,fill = TRUE)
}

# SVM
library(e1071)
for(i in 1:10){
  fold_test <- traindata[folds[[i]],] 
  fold_train <- traindata[-folds[[i]],] 
  fold_pre <- svm(form_clsm,data = fold_train,kernel ="radial",cost =200,gamma=0.08,probability = T)
  fold_predict <- predict(fold_pre,probability = T,newdata=fold_test) 
  fold_predict_p <- attr(fold_predict,"probabilities")
  fold_acc <- multiClassSummary(data.frame(obs = fold_test$sp,pred = fold_predict),lev = levels(fold_test$sp) )
  fold_auc <- multiclass.roc(response = fold_test$sp,predictor = fold_predict_p)
  cat("K:",i,'acc:',fold_acc[1],'auc:',fold_auc$auc,file = "file path",append = TRUE,fill = TRUE)
}

# training model
# RF
library(randomForest)
library(tidyverse)
library(skimr)
library(DataExplorer)
library(caret)
library(pROC)
FishData <- read.csv('path',header = T)
for (i in c(1)) {FishData[,i] <- factor(FishData[,i])}
set.seed(34)
trains <- createDataPartition(y = FishData$sp,p = 0.70,list = F)
traindata <-FishData[trains,]
testdata <- FishData[-trains,]
form_clsm <-as.formula(
  paste0("sp ~ ", paste(colnames(traindata)[2:20], collapse =" + "))
)

set.seed(34)
fit_rf_clsm <- randomForest(
  form_clsm,
  data = traindata,
  ntree = 100, 
  mtry = 4,
  importance = T,
  cost = 2,gamma =0.025
)
# training set
trainpredprob  <- predict(fit_rf_clsm, newdata = traindata, type = "prob")
multiclass.roc(response = traindata$sp,predictor = trainpredprob) 
trainpredlab <- predict(fit_rf_clsm, newdata = traindata, type = "class")
confusionMatrix(data = trainpredlab, 
                reference = traindata$sp, 
                mode = 'everything')
multiClassSummary(data.frame(obs = traindata$sp,
                             pred = trainpredlab),
                  lev = levels(traindata$sp) )
# test set
testpredprob<-predict(fit_rf_clsm,newdata = testdata,type = "prob")
multiclass.roc(response = testdata$sp,predictor = testpredprob)
testpredlab <- predict(fit_rf_clsm, newdata = testdata, type = "class")
RFTestMatrix <- confusionMatrix(data = testpredlab,
                                reference = testdata$sp,
                                mode = 'everything')
multiClassSummary(data.frame(obs = testdata$sp,
                             pred = testpredlab),
                  lev = levels(testdata$sp) )

# LDA
library(tidyverse)
library(caret)
library(cowplot)
library(MASS)
library(pROC)
FishData <- read.csv("path",header = T)
for (i in c(1)) {FishData[,i] <- factor(FishData[,i])}
set.seed(34)
trains <- createDataPartition(y = FishData$sp,p = 0.70,list = F)
traindata <-FishData[trains,]
testdata <- FishData[-trains,]

colnames(FishData) 
form_clsm <-as.formula(
  paste0("sp ~ ", paste(colnames(traindata)[2:length(traindata)], collapse =" + "))
)
form_clsm

set.seed(34)
fit_lda_clsm <- lda(
  form_clsm,
  data = traindata,
  method = "t",
)

# training set
trainpredprob  <- predict(fit_lda_clsm, newdata = traindata, type = "prob")
multiclass.roc(response = traindata$sp,predictor = trainpredprob$posterior)
trainpredlab <- predict(fit_lda_clsm, newdata = traindata, type = "class")
confusionMatrix(data = trainpredlab$class,
                reference = traindata$sp, 
                mode = 'everything')
multiClassSummary(data.frame(obs = traindata$sp,
                             pred = trainpredlab$class),
                  lev = levels(traindata$sp) )

# test set
testpredprob<-predict(fit_lda_clsm,newdata = testdata,
                      type = "prob")
multiclass.roc(response = testdata$sp,predictor = testpredprob$posterior)
testpredlab <- predict(fit_lda_clsm, newdata = testdata, type = "class")
ldaTestMatrix <- confusionMatrix(data = testpredlab$class, 
                                 reference = testdata$sp, 
                                 mode = 'everything')
multiClassSummary(data.frame(obs = testdata$sp,
                             pred = testpredlab$class),
                  lev = levels(testdata$sp) )

# QDA
library(tidyverse)
library(caret)
library(cowplot)
library(MASS)
FishData <- read.csv("path",header = T)
for (i in c(1)) {FishData[,i] <- factor(FishData[,i])}
set.seed(20)
trains <- createDataPartition(y = FishData$sp,p = 0.70,list = F)
traindata <-FishData[trains,]
testdata <- FishData[-trains,]
form_clsm <-as.formula(
  paste0("sp ~ ", paste(colnames(traindata)[2:length(traindata)], collapse =" + "))
)
set.seed(34) 
fit_qda_clsm <- qda(
  form_clsm,
  data = traindata,
  method = "t",
)
# training data
trainpredprob  <- predict(fit_qda_clsm, newdata = traindata, type = "prob")
multiclass.roc(response = traindata$sp,predictor = trainpredprob$posterior)
trainpredlab <- predict(fit_qda_clsm, newdata = traindata, type = "class")
confusionMatrix(data = trainpredlab$class,
                reference = traindata$sp,
                mode = 'everything')
multiClassSummary(data.frame(obs = traindata$sp,
                             pred = trainpredlab$class),
                  lev = levels(traindata$sp) )

# test data
testpredprob<-predict(fit_qda_clsm,newdata = testdata,
                      type = "prob")
multiclass.roc(response = testdata$sp,predictor = testpredprob$posterior)
testpredlab <- predict(fit_qda_clsm, newdata = testdata, type = "class")
QDATestMatrix <- confusionMatrix(data = testpredlab$class,
                                 reference = testdata$sp,
                                 mode = 'everything')
multiClassSummary(data.frame(obs = testdata$sp,
                             pred = testpredlab$class),
                  lev = levels(testdata$sp) )

# SVM
library(e1071)
library(tidyverse)
library(skimr)
library(pROC)
library(caret)
fishdata <- read.csv("path") 
for (i in c(1)) {fishdata[,i] <- factor(fishdata[,i])}
set.seed(34)
trains <- createDataPartition(y = fishdata$sp,p = 0.70,list = F)
traindata <-fishdata[trains,]
traindata[2:length(fishdata)] <- scale(traindata[2:length(fishdata)],center = T,scale = T)
testdata <- fishdata[-trains,]
testdata[2:length(fishdata)] <- scale(testdata[2:length(fishdata)],center = T,scale = T)

form_clsm <-as.formula(
  paste("sp ~ ", paste(colnames(traindata)[2:length(fishdata)], collapse =" + "))
)

fit_svm_clsm <- svm(form_clsm,
                    data = traindata,
                    kernel ="radial",
                    cost =200,
                    gamma=0.03,
                    probability = T)

# training set
trainpred <- predict(fit_svm_clsm, newdata = traindata, probability = T)
trainpredprob <- attr(trainpred,"probabilities")
multiclass.roc(response = traindata$sp,predictor = trainpredprob)
confusionMatrix(data = trainpred,
                reference = traindata$sp,
                mode = 'everything')
multiClassSummary(data.frame(obs = traindata$sp,
                             pred = trainpred),
                  lev = levels(traindata$sp) )

#test set
testpred<-predict(fit_svm_clsm,newdata = testdata,
                  probability = T)
testpredprob<- attr(testpred, 'probabilities')
multiclass.roc(response = testdata$sp,predictor = testpredprob)
SVMTestMatrix <- confusionMatrix(data = testpred,
                                 reference = testdata$sp,
                                 mode = 'everything')
multiClassSummary(data.frame(obs = testdata$sp,
                             pred = testpred),
                  lev = levels(testdata$sp))
