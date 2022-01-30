# testing parallelization

library(e1071)
library(tidyverse)
#library(adabag)
library(parallel)

#########################
# Load and prepare data #
#########################

load(file = '~/ass2/MNIST_for_assignment.Rdata')

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
sample.vec <- c(c(50,100), c(2:10)*1e2, c(2:5)*1e3, c(7500, 10000))

boost.results <- function(n0){
  cat(paste0('running: ', n0, '\n'))
  #library(adabag)
  # number of training samples
  num_train <- n0   # 60000 for full data set 
  
  # select subset for training
  train_data <- mnist.train[1:num_train,]
  train_label <- factor(mnist.train_label[1:num_train])
  X <- cbind(train_data, train_label)
  
  # Init timer
  
  
  # Train Boosting
  t1 <- proc.time()
  
  S <- adabag::boosting(train_label ~., data=X)
  
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
    'method' = 'ada',
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

clusterExport(cl, varlist = c('mnist.train','mnist.train_label','test_data','test_label'))

cat(' ***  starting... ***')
t1 <- proc.time()
x <- parLapply(cl,sample.vec, boost.results)
print(proc.time()-t1)
stopCluster(cl)
df.results <- data.frame()

for(i in c(1:length(x))){
  df.results <- bind_rows(df.results, x[[i]])
}

save(list = c('df.results'), file = '~/ass2/Boosting_results_test.Rdata')
