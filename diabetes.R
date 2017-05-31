library(caret)
library(nnet)
diabetes <- read.csv("C:/lisp/emacshome/test/Diabetes.csv")
str(diabetes)
library(AppliedPredictiveModeling)
transparentTheme(trans = .4)
apply(dataset,2,function(x) sum(is.na(x)))
dataset <- diabetes
featurePlot(x = dataset[,1:8], y = dataset$Classify, plot="pairs",auto.key=list(colums = 2))
set.seed(998)
inTraining <- createDataPartition(dataset$Classify, p = .75, list = FALSE)
training <- dataset[inTraining,]
testing <- dataset[-inTraining,]

x_test <- subset(testing, select = -Classify)
y_test <- testing$Classify

training$Classify = factor(training$Classify)
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  repeats = 10)

nnetGrid <- expand.grid(size=c(10), decay=c(0,0.0001, 0.001,0.01,0.004))

model_nnet <- train(Classify ~ ., training, method='nnet', 
                             trace = FALSE, 
                             metric = "Accuracy", 
                             preProc = c("center", "scale"), 
                             trControl = fitControl,
                             tuneGrid = nnetGrid)

#Neural Network 
#
#576 samples
#8 predictor
#2 classes: '0', '1' 

#Pre-processing: centered (8), scaled (8) 
#Resampling: Cross-Validated (3 fold, repeated 3 times) 
#Summary of sample sizes: 384, 384, 384, 384, 385, 383, ... 
#Resampling results across tuning parameters:
  
#  size  decay  Accuracy   Kappa    
#1     0e+00  0.7556967  0.4598711
#1     1e-04  0.7441075  0.4561493
#1     1e-01  0.7644287  0.4702265
#3     0e+00  0.7124167  0.3810135
#3     1e-04  0.7297542  0.4338652
#3     1e-01  0.7528273  0.4520066
#5     0e+00  0.7089085  0.3709162
#5     1e-04  0.7007673  0.3449034
#5     1e-01  0.7267943  0.3931642

prediction <- predict(model_nnet, testing) 
conf <- table(prediction, y_test) 
confusionMatrix(conf)
#Confusion Matrix and Statistics

#y_test
#prediction   0   1
#0 107  24
#1  22  39

#Accuracy : 0.7604          
#95% CI : (0.6937, 0.8189)
#No Information Rate : 0.6719          
#P-Value [Acc > NIR] : 0.00474         

#Kappa : 0.4522          
#Mcnemar's Test P-Value : 0.88278         

model_avnnet <- train(Classify ~ ., training, method='avNNet', 
                              trace = FALSE, 
                              metric = "Accuracy", 
                               preProc = c("center", "scale")
              )
#0e+00  0.7486255  0.4387122
prediction2 <- predict(model_avnnet, testing) 
conf2 <- table(prediction2, y_test) 
confusionMatrix(conf2)

#Confusion Matrix and Statistics

#y_test
##prediction   0   1
#0 114  28
#1  15  35

#Accuracy : 0.776           
#95% CI : (0.7104, 0.8329)
#No Information Rate : 0.6719          
#P-Value [Acc > NIR] : 0.001015       
library(elmNN)
library(MASS)

model_elm <- train(Classify ~ ., training, method='elm', 
                                trace = FALSE, 
                                metric = "Accuracy", 
                                preProc = c("center", "scale")
                 )
#purelin  0.6987133  0.33431891

prediction3 <- predict(model_elm, testing) 
conf3 <- table(prediction3, y_test) 
confusionMatrix(conf3)

#Confusion Matrix and Statistics

#y_test
#prediction3   0   1
#0 113  28
#1  16  35

#Accuracy : 0.7708          
#95% CI : (0.7048, 0.8283)
#No Information Rate : 0.6719          
#P-Value [Acc > NIR] : 0.001744  

# ensemble using Stacking method
library(caretEnsemble)
seed = 100
levels(training$Classify) = c('healthy','diabete')
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions="final", classProbs=TRUE)
algorithmList <- c('nnet', 'elm')
set.seed(seed)
models <- caretList(Classify~., data=training, trControl=control, methodList=algorithmList,
                     trace= FALSE, metric="Accuracy")
#models <- caretList(Classify~., data=training, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)

#Models: nnet, elm 
#Number of resamples: 30 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#nnet 0.5964912 0.6506352 0.6957048 0.7040734 0.7586207 0.8275862    0
#elm  0.5438596 0.6422414 0.6896552 0.6869530 0.7241379 0.8103448    0

#Kappa 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#nnet  0.0000000 0.1868862 0.2697815 0.3065034 0.4575124 0.6091644    0
#elm  -0.1312977 0.1518316 0.2373660 0.2416588 0.3334853 0.5301915    0
dotplot(results)
modelCor(results) # low correlation
#      nnet       elm
#nnet 1.0000000 0.3843745
#elm  0.3843745 1.0000000

stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions="final", classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

########
algorithmList <- c('nnet', 'knn')
set.seed(seed)
models <- caretList(Classify~., data=training, trControl=control, methodList=algorithmList,
                    trace= FALSE, metric="Accuracy")
#models <- caretList(Classify~., data=training, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
#Models: nnet, knn 
#Number of resamples: 30 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#nnet 0.5263158 0.6666667 0.6982759 0.7093829 0.7758621 0.8275862    0
#knn  0.5964912 0.6926800 0.7241379 0.7251303 0.7543860 0.8448276    0

#Kappa 
#            Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#nnet -0.07697691 0.2298156 0.3271994 0.3387396 0.5056913 0.6267696    0
#knn   0.06021505 0.3097077 0.3959777 0.3886980 0.4481328 0.6605982    0

stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions="final", classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
#Accuracy   Kappa    
#0.7266616  0.3790731

# using Stochastic Gradient Boosting 
library("gbm")
gbm_ensemble <- caretStack(
  models,
  method="gbm",
  verbose=FALSE,
  metric="Accuracy",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)

#Ensemble results:
#  Stochastic Gradient Boosting 

#1728 samples
#2 predictor
#2 classes: 'healthy', 'diabete' 

#No pre-processing
#Resampling: Bootstrapped (10 reps) 
#Summary of sample sizes: 1728, 1728, 1728, 1728, 1728, 1728, ... 
#Resampling results across tuning parameters:
  
#  interaction.depth  n.trees  ROC        Sens       Spec     
#1                   50      0.7975256  0.8284758  0.5692841
#1                  100      0.7980054  0.8323768  0.5680051
#1                  150      0.7970357  0.8351733  0.5599820
#2                   50      0.7958735  0.8362659  0.5595255
#2                  100      0.7915628  0.8333061  0.5431705
#2                  150      0.7868277  0.8289743  0.5431971
#3                   50      0.7919394  0.8291357  0.5553183
#3                  100      0.7862822  0.8253056  0.5387377
#3                  150      0.7806771  0.8244493  0.5321768

#Tuning parameter 'shrinkage' was held constant at a value of 0.1
#Tuning parameter 'n.minobsinnode' was held constant at a value of 10
#ROC was used to select the optimal model using  the largest value.
#The final values used for the model were n.trees = 100, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

ensemble <- predict(gbm_ensemble, newdata=testing, type="prob")

library("caTools")
colAUC(ensemble, y_test) #0 vs. 1 0.7895903
