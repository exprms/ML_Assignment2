#
#
#  CIFAR
#

library(keras)
library(tidyverse)

load('R_CIFAR.Rdata')

# Select subset for testing
num_test  <-  500
test_data <-  cifar.test[1:num_test,,,]
test_label <-  cifar.test_label[1:num_test]

# Create categorical labels
test_labelc <-  to_categorical(test_label, num_classes = 10)

# Tuning Grid
tune.grid <- expand.grid(
  num_train = c(5000,10000),
  batch_size = c(32,64),
  num_epochs = c(5,10),
  num_of_units1 = c(128,256),
  num_of_units2 = 128
)

# Result frame
df.results <- data.frame()

# Training and validation LOOP
for (i in c(1:nrow(tune.grid))){
  # Training Data
  num_train <- tune.grid$num_train[i]
  
  train_data <-  cifar.train[1:num_train,,,]
  train_label <-  cifar.train_label[1:num_train]
  train_labelc <-  to_categorical(train_label, num_classes = 10)
  
  # settings
  batch_size <-  tune.grid$batch_size[i]
  num_epochs <-  tune.grid$num_epochs[i]
  num_of_units1 <-  tune.grid$num_of_units1[i]
  
  # Init model
  CNN <- keras_model_sequential()
  
  # Define network architecture
  CNN %>%
    # First BLOCK
    layer_conv_2d( filter = 32, 
                   kernel_size = c(3,3), 
                   padding = "same", 
                   input_shape = c(32, 32, 3)) %>%
    layer_activation("relu") %>%
    
    layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%

    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    
    #layer_dropout(0.25) %>%
    
    # Flatten input into feature vector and feed into dense layer
    layer_flatten() %>%
    layer_dense(num_of_units1) %>%
    layer_activation("relu") %>%
    #layer_dropout(0.5) %>%
    
    #layer_dense(num_of_units) %>%
    #layer_activation("relu") %>%
    #layer_dropout(0.5) %>%
    
    # Outputs from dense layer are projected onto 10 unit output layer
    layer_dense(10) %>%
    layer_activation("softmax")
  
  
  # Set parameters for optimizer
  opt <- optimizer_rmsprop(learning_rate = 0.0001, decay = 1e-6)
  
  # Compile network
  CNN %>% compile(
    loss = "categorical_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
  )
  
  ################################
  # Train and run classification #
  ################################
  
  # TRAINING
  t1 = proc.time()
  
  CNN %>% fit(
    train_data, train_labelc,
    batch_size = batch_size,
    epochs = num_epochs,
    shuffle = TRUE
  )
  
  t2 = proc.time()
  t = t2-t1
  
  # Eval CNN on training data
  # result <- CNN %>% evaluate(train_data, train_labelc)
  # success =result$acc[1]*100
  
  # Eval CNN on test data
  result <- CNN %>% evaluate(test_data, test_labelc)
  
  tmp <- data.frame(
    'method' = 'NN',
    'train.size' = tune.grid$num_train[i],
    'epochs' = tune.grid$num_epochs[i],
    'batchsize' = tune.grid$batch_size[i],
    'hidden1' = tune.grid$num_of_units1[i],
    'hidden2' = NA,
    'hidden3' = NA,
    'activation' = 'relu',
    'success.test' = result$acc,
    'elapsed.training' = unname(t['elapsed'])
  )
  
  df.results <- bind_rows(
    df.results,
    tmp
  )
}
save(list=c('df.results'), file = 'cifar_results1.Rdata')

## 3 BLOCKS
# Tuning Grid
tune.grid <- expand.grid(
  num_train = 10000,
  batch_size = 32,
  num_epochs = c(5,10),
  num_of_units1 = 256,
  num_of_units2 = 256
)
# Training and validation LOOP
for (i in c(1:nrow(tune.grid))){
  # Training Data
  num_train <- tune.grid$num_train[i]
  
  train_data <-  cifar.train[1:num_train,,,]
  train_label <-  cifar.train_label[1:num_train]
  train_labelc <-  to_categorical(train_label, num_classes = 10)
  
  # settings
  batch_size <-  tune.grid$batch_size[i]
  num_epochs <-  tune.grid$num_epochs[i]
  num_of_units1 <-  tune.grid$num_of_units1[i]
  num_of_units2 <-  tune.grid$num_of_units2[i]
  
  # Init model
  CNN <- keras_model_sequential()
  
  # Define network architecture
  CNN %>%
    # First BLOCK
    layer_conv_2d( filter = 32, 
                   kernel_size = c(3,3), 
                   padding = "same", 
                   input_shape = c(32, 32, 3)) %>%
    layer_activation("relu") %>%
    
    layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%
    
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    
    layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%
    
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    
    layer_conv_2d(filter = 128, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%
    
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    
    layer_dropout(0.25) %>%
    
    # Flatten input into feature vector and feed into dense layer
    layer_flatten() %>%
    layer_dense(num_of_units1) %>%
    layer_activation("relu") %>%
    layer_dropout(0.5) %>%
    
    layer_dense(num_of_units2) %>%
    layer_activation("relu") %>%
    layer_dropout(0.5) %>%
    
    # Outputs from dense layer are projected onto 10 unit output layer
    layer_dense(10) %>%
    layer_activation("softmax")
  
  
  # Set parameters for optimizer
  opt <- optimizer_rmsprop(learning_rate = 0.0001, decay = 1e-6)
  
  # Compile network
  CNN %>% compile(
    loss = "categorical_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
  )
  
  ################################
  # Train and run classification #
  ################################
  
  # TRAINING
  t1 = proc.time()
  
  CNN %>% fit(
    train_data, train_labelc,
    batch_size = batch_size,
    epochs = num_epochs,
    shuffle = TRUE
  )
  
  t2 = proc.time()
  t = t2-t1
  
  # Eval CNN on training data
  # result <- CNN %>% evaluate(train_data, train_labelc)
  # success =result$acc[1]*100
  
  # Eval CNN on test data
  result <- CNN %>% evaluate(test_data, test_labelc)
  
  tmp <- data.frame(
    'method' = 'NN - 3 Blocks - dropout',
    'train.size' = tune.grid$num_train[i],
    'epochs' = tune.grid$num_epochs[i],
    'batchsize' = tune.grid$batch_size[i],
    'hidden1' = tune.grid$num_of_units1[i],
    'hidden2' = tune.grid$num_of_units2[i],
    'hidden3' = NA,
    'activation' = 'relu',
    'success.test' = result$acc,
    'elapsed.training' = unname(t['elapsed'])
  )
  
  df.results <- bind_rows(
    df.results,
    tmp
  )
}

## 2 BLOCKS TEIL 2
# Tuning Grid
tune.grid <- expand.grid(
  num_train = 10000,
  batch_size = 32,
  num_epochs = c(5,10),
  num_of_units1 = 256,
  num_of_units2 = 256
)
# Training and validation LOOP
for (i in c(1:nrow(tune.grid))){
  # Training Data
  num_train <- tune.grid$num_train[i]
  
  train_data <-  cifar.train[1:num_train,,,]
  train_label <-  cifar.train_label[1:num_train]
  train_labelc <-  to_categorical(train_label, num_classes = 10)
  
  # settings
  batch_size <-  tune.grid$batch_size[i]
  num_epochs <-  tune.grid$num_epochs[i]
  num_of_units1 <-  tune.grid$num_of_units1[i]
  num_of_units2 <-  tune.grid$num_of_units2[i]
  
  # Init model
  CNN <- keras_model_sequential()
  
  # Define network architecture
  CNN %>%
    # First BLOCK
    layer_conv_2d( filter = 32, 
                   kernel_size = c(3,3), 
                   padding = "same", 
                   input_shape = c(32, 32, 3)) %>%
    layer_activation("relu") %>%
    
    layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%
    
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    
    layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%
    
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    
    layer_conv_2d(filter = 128, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%
    
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    
    layer_dropout(0.25) %>%
    
    # Flatten input into feature vector and feed into dense layer
    layer_flatten() %>%
    layer_dense(num_of_units1) %>%
    layer_activation("relu") %>%
    layer_dropout(0.5) %>%
    
    #layer_dense(num_of_units2) %>%
    #layer_activation("relu") %>%
    #layer_dropout(0.5) %>%
    
    # Outputs from dense layer are projected onto 10 unit output layer
    layer_dense(10) %>%
    layer_activation("softmax")
  
  
  # Set parameters for optimizer
  opt <- optimizer_rmsprop(learning_rate = 0.0001, decay = 1e-6)
  
  # Compile network
  CNN %>% compile(
    loss = "categorical_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
  )
  
  ################################
  # Train and run classification #
  ################################
  
  # TRAINING
  t1 = proc.time()
  
  CNN %>% fit(
    train_data, train_labelc,
    batch_size = batch_size,
    epochs = num_epochs,
    shuffle = TRUE
  )
  
  t2 = proc.time()
  t = t2-t1
  
  # Eval CNN on training data
  # result <- CNN %>% evaluate(train_data, train_labelc)
  # success =result$acc[1]*100
  
  # Eval CNN on test data
  result <- CNN %>% evaluate(test_data, test_labelc)
  
  tmp <- data.frame(
    'method' = 'NN - 3 Blocks',
    'train.size' = tune.grid$num_train[i],
    'epochs' = tune.grid$num_epochs[i],
    'batchsize' = tune.grid$batch_size[i],
    'hidden1' = tune.grid$num_of_units1[i],
    'hidden2' = tune.grid$num_of_units2[i],
    'hidden3' = NA,
    'activation' = 'relu',
    'success.test' = result$acc,
    'elapsed.training' = unname(t['elapsed'])
  )
  
  df.results <- bind_rows(
    df.results,
    tmp
  )
}

save(list=c('df.results'), file = 'cifar_results2.Rdata')
