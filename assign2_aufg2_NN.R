#
##
#  aufgabe 1 - NN
#
# 


library(nnet)
library(tidyverse)
#library(keras)
library(deepnet)
#library(adabag)

#########################
# Load and prepare data #
#########################

load(file = 'MNIST_for_assignment.Rdata')

# bossting wants a data frame
# mnist.train <- data.frame(mnist.train)

# number of test samples
num_test  <-  500   # 10000 for full data set

# select subset for testing
test_data  <- mnist.test[1:num_test,]
test_label <- factor(mnist.test_label[1:num_test])
cil_te     <- class.ind(test_label)  # Indikatormatrix

# #####################################
# Classification Neural Network       #
# #####################################

df.results <- data.frame()
sample.vec <- c(c(2:5)*1e3, c(1:6)*1e4)

# Tuning Grid:
tunegrid <- expand.grid(epochs = c(3,5,10,20),
                        activation= c('sigm','tanh'),
                        hidden1 = c(10,28,56))

T0 <- proc.time()

for(n0 in sample.vec){
  
  # number of training samples
  num_train <- n0   # 60000 for full data set 
  
  # select subset for training
  train_data <- mnist.train[1:num_train,]
  train_label <- factor(mnist.train_label[1:num_train])
  cil_tr <- class.ind(train_label)
  
  for (j in c(1:nrow(tunegrid))){
    
    cat(paste0('samplesize = ', n0, '\n'))
    
    # Training NN Model
    t1 <- proc.time()
    S <- nn.train(
      x = train_data, 
      y = cil_tr,
      hidden = c(tunegrid$hidden1[j]),
      numepochs = tunegrid$epochs[j],
      activationfun = tunegrid$activation[j])
    t2 <- proc.time()
    training.time <- t2-t1
    
    # Eval NN on training data
    t1 <- proc.time()
    pr_tr <- nn.predict(S, train_data)
    success.train <- sum(max.col(pr_tr)==max.col(cil_tr))/length(train_label)*100
    
    # Eval NN on test data
    pr_te <- nn.predict(S, test_data)
    success.test <- sum(max.col(pr_te)==max.col(cil_te))/length(test_label)*100
    
    t2 <- proc.time()
    testing.time <- t2-t1
    
    # Write to result frame
    tmp <- data.frame(
      'method' = 'NN',
      'train.size' = n0,
      'epochs' = tunegrid$epochs[j],
      'hidden1' = tunegrid$hidden1[j],
      'activation' = tunegrid$activation[j],
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
}

total.time <- proc.time()-T0
print(total.time)

save(list = c('df.results'), file = 'NN_results_parmeter.Rdata')

