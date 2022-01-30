#
##
#  aufgabe 1


library(e1071)
library(tidyverse)
library(adabag)

#########################
# Load and prepare data #
#########################

load(file = 'MNIST_for_assignment.Rdata')

# bossting wants a data frame
mnist.train <- data.frame(mnist.train)

# number of test samples
num_test  <-  500   # 10000 for full data set

# select subset for testing
test_data  <- data.frame(mnist.test[1:num_test,])
test_label <- mnist.test_label[1:num_test]

# #####################################
# Classification Boosting - standard  #
# #####################################

df.results <- data.frame()
sample.vec <- c(c(50,100), c(2:10)*1e2, c(2:10)*1e3)

T0 <- proc.time()

for(n0 in sample.vec){
  
  cat(paste0('samplesize = ', n0))
  
  # number of training samples
  num_train <- n0   # 60000 for full data set 
  
  # select subset for training
  train_data <- mnist.train[1:num_train,]
  train_label <- factor(mnist.train_label[1:num_train])
  X <- cbind(train_data, train_label)
  
  # Init timer
  
  
  # Train Boosting
  t1 <- proc.time()
  
  S <- boosting(train_label ~., data=X)
  
  t2 <- proc.time()
  training.time <- t2-t1
  
  # Eval Boosting on training data
  t1 <- proc.time()
  
  pr_tr <- predict(S, train_data)
  success.train <- sum(pr_tr$class==factor(train_label))/length(train_label)*100
  
  # Eval SVM on test data
  pr_te <- predict(S, test_data)
  success.test <- sum(pr_te$class==factor(test_label))/length(test_label)*100
  
  t2 <- proc.time()
  testing.time <- t2-t1
  
  
  tmp <- data.frame(
    'method' = 'RF',
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

save(list = c('df.results'), file = 'Boosting_results.Rdata')

