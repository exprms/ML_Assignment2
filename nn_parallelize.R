# testing parallelization

library(e1071)
library(tidyverse)
#library(adabag)
library(parallel)
library(nnet)
library(deepnet)

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
cil_te     <- class.ind(test_label)  # Indikatormatrix

# #####################################
# Classification Neural Network       #
# #####################################

df.results <- data.frame()
sample.vec <- c(c(2:5)*1e3, c(7500, 10000))

nn.results <- function(n0){
  # number of training samples
  num_train <- n0   # 60000 for full data set 
  
  # select subset for training
  train_data <- mnist.train[1:num_train,]
  train_label <- factor(mnist.train_label[1:num_train])
  cil_tr <- class.ind(train_label)
  
  # Training NN Model
  t1 <- proc.time()
  S <- deepnet::nn.train(
    x = train_data, 
    y = cil_tr)
  t2 <- proc.time()
  training.time <- t2-t1
  
  # Eval NN on training data
  t1 <- proc.time()
  pr_tr <- nn.predict(S, train_data)
  success.train <- sum(max.col(pr_tr)==max.col(cil_tr))/length(train_label)*100
  
  # Eval NN on test data
  pr_te <- deepnet::nn.predict(S, test_data)
  success.test <- sum(max.col(pr_te)==max.col(cil_te))/length(test_label)*100
  
  t2 <- proc.time()
  testing.time <- t2-t1
  
  # Write to result frame
  tmp <- data.frame(
    'method' = 'NN',
    'train.size' = n0,
    'success.train' = success.train,
    'success.test' = success.test,
    'elapsed.training' = unname(training.time['elapsed']),
    'elapsed.testing' = unname(testing.time['elapsed'])
  )
  
  return(tmp)
}

n_cores <- detectCores(logical = FALSE)
cl <- makeCluster(n_cores-1)

clusterExport(cl, varlist = c('mnist.train','mnist.train_label','test_data','test_label','cil_te'))

cat(' ***  starting... ***')
t1 <- proc.time()
x <- parLapply(cl,sample.vec, nn.results)
print(proc.time()-t1)
stopCluster(cl)
df.results <- data.frame()

for(i in c(1:length(x))){
  df.results <- bind_rows(df.results, x[[i]])
}

save(list = c('df.results'), file = 'VM_NN_results.Rdata')
