#
#
#   PREPARE MNIST DATA FOR ASSIGNMENT
#

library(keras)

#########################
# Load and prepare data #
#########################

# Load MNIST data
mnist <- dataset_mnist()

# Assign train and test data+labels
mnist.train  <- mnist$train$x
mnist.train_label <- mnist$train$y

mnist.test  <- mnist$test$x
mnist.test_label <- mnist$test$y

# Reshape images to vectors
mnist.train <- array_reshape(mnist.train, c(nrow(mnist.train), 784))
mnist.test  <- array_reshape(mnist.test, c(nrow(mnist.test), 784))

# Rescale data to range [0,1]
mnist.train <- mnist.train / 255
mnist.test <- mnist.test / 255

save(
  list = c('mnist.test', 'mnist.train', 'mnist.test_label', 'mnist.train_label'), 
  file = 'MNIST_for_assignment.Rdata'
)
