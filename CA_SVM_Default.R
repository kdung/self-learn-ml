#=====================================================================
###  SVM with Default dataset
#=====================================================================

#initialize
library(ISLR)
library("e1071")
data("Default");
summary(Default)
dat = Default
attach(dat)

# sampling 80:20
set.seed(1)
sample <- sample.int(n = nrow(dat), size = floor(.8*nrow(dat)), replace = F)
train <- dat[sample,]
test <- dat[-sample,]
summary(test)
summary(train)
y_test <- test$default
x_test <- subset(test, select = -default)

# split data set
# combination 1: income and balance
train_1 <- subset(train, select = -student)
test_1 <- subset(test, select = -student)
x_test_1 <- subset(test_1, select = -default)

# combination 2: student and income
test_2 <- subset(test, select = -balance)
train_2 <- subset(train, select = -balance)
x_test_2 <- subset(test_2, select = -default)

# combination 3: student and balance
test_3 <- subset(test, select = -income)
train_3 <- subset(train, select = -income)
x_test_3 <- subset(test_3, select = -default)


#===================== RADIAL KERNEL: combination 1: income and balance ==============
# 
svm_tune <- tune.svm(default ~., data = train_1, cost = c(2^-3,2^-1,2^1,2^3,2^5,2^7,2^8,2^10))
summary(svm_tune) #  cost = 256; best performance: 0.0275

svm_tune <- tune.svm(default ~., data = train_3, cost = c(2^1,2^3,2^5,2^7,2^8,2^10), gamma = c(2^-7,2^-5,2^-3,2^-1,2^3)) 
# gamma = 2, cost = 256; best performance: 0.026625 
svm <- svm(default ~ ., data = train_1, kernel = "radial", cost = 256, gamma = 2)
summary(svm) # Number of Support Vectors:  683, ( 461 222 )
pred <- predict(svm, x_test_1)
table(pred, y_test)
#pred    No  Yes
#No  1927   43
#Yes   10   20
prec = 20/(20+10) # prec = 0.6666, 
rec = 20/(20+43) #0.3174
f1 = 2 * prec * rec / (prec + rec) # 0.43 
acc = (21+1922)/2000 # acc = 0.9735

# =============================== RADIAL KERNEL combination 2: student and income ==========

# tuning
svm_tune <- tune.svm(default ~., data = train_2, cost = c(2^-3,2^-1,2^1,2^3,2^5,2^10))
# sampling method: 10-fold cross validation 
# best parameters: cost = 0.125, best performance: 0.0335
# tune both C and gamma again with smaller range of C
svm_tune <- tune.svm(default ~., data = train_3, cost = c(2^-5,2^-3,2^-1,2^1,2^3), gamma = c(2^-7,2^-5,2^-3,2^-1,2^3)) 
# best parameters: gamma = 0.0078125, cost = 0.125, best performance: 0.0335 
summary(svm_tune) # gamma = 0.0078125 best performance: 0.0335 

# train svm again with new parameters
svm <- svm(default ~ ., data = train_2, kernel = "radial", cost = 0.125, gamma = 0.0078125)
summary(svm)  # Number of Support Vectors:  550  ( 282 268 )
pred <- predict(svm, x_test_2)
table(pred, y_test)
#pred    No  Yes
#No     1935   65
#Yes     0    0
prec = 0
rec = 0
f1 = 2 * prec * rec / (prec + rec)# = 0
acc = 1935/2000 #0.9675

# =============================== RADIAL KERNEL combination 3: student and balance ==========

# tuning 
svm_tune <- tune.svm(default ~., data = train_3, cost = c(2^-3,2^-1,2^1,2^3,2^5,2^10)) 
# cost = 2 best performance: 0.029 
# sampling method: 10-fold cross validation 
# tune again with smaller range of C
svm_tune <- tune.svm(default ~., data = train_3, cost = c(2^-3,2^-1,2^1,2^3), gamma = c(2^-7,2^-5,2^-3,2^-1,2^3)) 
# cost = 2 gamma = 0.5 best performance: 0.02675 
summary(svm_tune) 

# train svm again with new parameters
svm <- svm(default ~ ., data = train_3, kernel = "radial", cost = 2, gamma = 0.5)
summary(svm)  # Number of Support Vectors:  518  ( 284 234 )
pred <- predict(svm, x_test_3)
table(pred, y_test)
#pred    No    Yes
#No     1924   46
#Yes     11    19
prec = 19/(19+11) # prec = 0.6333333
rec = 19/(19+46) #0.2923077
f1 = 2 * prec * rec / (prec + rec) # 0.4
acc = (1924+19)/2000 # acc = 0.9715

#======================== POLYNOMIAL combination 1: income and balance =================
svm.tune <- tune.svm(default~., data = train_1,  cost = 2^(2:5), kernel = "polynomial") 
summary(svm.tune) # cost = 4, best performance: 0.029125 
svm_poly <- svm(default ~ ., data = train_1, kernel = "polynomial", cost = 4)
summary(svm_poly) #Number of Support Vectors:  504, degree:  3, gamma:  0.5, coef.0:  0 
pred <- predict(svm_poly, x_test_1)
table(pred, y_test)
#No  1930   48
#Yes    7   15
prec = 15/(15+8) #0.6521739
rec = 15/(15+48) #0.2380952
f1 = 2 * prec * rec / (prec + rec) #0.3488372
acc = (15+1930)/2000 # 0.9725

#======================== POLYNOMIAL combination 2: student and income =================
svm.tune <- tune.svm(default~., data = train_2,  cost = 2^(2:5), kernel = "polynomial") 
summary(svm.tune) # cost = 4, best performance: 0.0335 
svm_poly <- svm(default ~ ., data = train_2, kernel = "polynomial", cost = 4)
summary(svm_poly) #Number of Support Vectors:  768 ( 500 268 ), degree:  3, gamma:  0.3333333, coef.0:  0 
pred <- predict(svm_poly, x_test_2)
table(pred, y_test)
#No  1935   65
#Yes    0   0
prec = 0 
rec = 0
f1 = 2 * prec * rec / (prec + rec) # 0
acc = (1935)/2000 # 0.9675

#======================== POLYNOMIAL combination 3: student and balance =================
svm.tune <- tune.svm(default~., data = train_3,  cost = 2^(2:5), kernel = "polynomial") 
summary(svm.tune) # cost = 4, best performance: 0.02675
svm_poly <- svm(default ~ ., data = train_3, kernel = "polynomial", cost = 4)
summary(svm_poly) #Number of Support Vectors:  477 ( 240 237 ), degree:  3, gamma:  0.3333333, coef.0:  0 
pred <- predict(svm_poly, x_test_3)
table(pred, y_test)
#No    1924  47
#Yes    11   18
prec = 18/(18+11) #0.6206897
rec = 18/(18+47) #0.2769231
f1 = 2 * prec * rec / (prec + rec) #0.3829787
acc = (18+1924)/2000 # 0.971


#=======================LINEAR KERNEL ================================================
svm <- svm(default ~ ., data = train, kernel = "linear", cost = 16)
#Number of Support Vectors:  529
pred <- predict(svm, x_test)
table(pred, y_test)
#pred    No  Yes
#No  1937   63
#Yes    0    0

#======================SIGMOID KERNEL=================================================
svm <- svm(default ~ ., data = train, kernel = "sigmoid", cost = 16)
summary(svm)
# SVM-Kernel:  sigmoid, cost:  16, gamma:  0.25, coef.0:  0, Number of Support Vectors:  415
pred <- predict(svm, x_test)
table(pred, y_test)
#pred    No  Yes
#No  1880   42
#Yes   57   21

