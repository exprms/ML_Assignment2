#
##
#  aufgabe 1


library(e1071)
library(tidyverse)
library(randomForest)

#########################
# Load and prepare data #
#########################

load(file = 'MNIST_for_assignment.Rdata')

# number of test samples
num_test  <-  500   # 10000 for full data set

# select subset for testing
test_data  <- mnist.test[1:num_test,]
test_label <- mnist.test_label[1:num_test]

# ##########################################
# Classification Random Forest - standard  #
# ##########################################

df.results <- data.frame()
sample.vec <- c(c(50,100), c(2:10)*1e2, c(2:10)*1e3)
n0 <- 1000

# Tuning Grid:
tunegrid <- expand.grid(mtry = c(4,8,28),
                        ntree= c(100,200,300,500),
                        nodesize=c(1,2,5,15))
T0 <- proc.time()
#for(n0 in sample.vec){
for(j in c(1:nrow(tunegrid))){
  cat(paste0(j, '\n'))
  num_train <- n0   # 60000 for full data set 
  
  # select subset for training
  train_data <- mnist.train[1:num_train,]
  train_label <- mnist.train_label[1:num_train]
  
  # Init timer
  t1 <- proc.time()
  
  # Train RF
  S <- randomForest(
    train_data, 
    factor(train_label), 
    ntree = tunegrid$ntree[j],
    mtry = tunegrid$mtry[j],
    nodesize = tunegrid$nodesize[j]
    )
  
  t2 <- proc.time()
  training.time <- t2-t1
  
  # Eval RF on training data
  t1 <- proc.time()
  
  pr_tr <- predict(S, train_data)
  success.train <- sum(pr_tr==factor(train_label))/length(train_label)*100
  
  # Eval SVM on test data
  pr_te <- predict(S, test_data)
  success.test <- sum(pr_te==factor(test_label))/length(test_label)*100
  
  # End time, calculate elapsed time
  t2 <- proc.time()
  testing.time <- t2-t1
  
  tmp <- data.frame(
    'method' = 'RF',
    'mtry' = tunegrid$mtry[j],
    'ntree' = tunegrid$ntree[j],
    'nodesize' = tunegrid$nodesize[j],
    'train.size' = n0,
    'success.train' = success.train,
    'success.test' = success.test,
    'elapsed.training' = unname(training.time['elapsed']),
	'elapsed.testing' = unname(testing.time['elapsed'])
  )
  
  df.results <- bind_rows(
    df.results,
    tmp
  )
}

total.time <- proc.time()-T0
print(total.time)

save(list = c('df.results'), file = 'RF_results_parameter.Rdata')


df.results %>% 
  ggplot(aes(x=elapsed.training, 
             y=success.test, col=factor(ntree)))+
  geom_point(aes(pch=factor(nodesize)))
