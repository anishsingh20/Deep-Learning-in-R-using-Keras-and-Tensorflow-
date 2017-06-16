#Training a simple deep CNN Model on cifar10 small image dataset
#Image Recognition
#Classify images amongst 10 classes of airplane, automobile, bird, cat
#deer, dog, frog, horse, ship, and truck.

library(keras)


cifar<-dataset_cifar10()
#Details of the Image Data
?dataset_cifar10

#TRAINING DATA
train_x<-cifar$train$x/255
#convert a vector class to binary class matrix
train_y<-to_categorical(cifar$train$y,num_classes = 10)


#TEST DATA
test_x<-cifar$test$x/255
test_y<-to_categorical(cifar$test$y,num_classes=10)

#checking the dimentions
dim(train_x)

cat("No of training samples\t",dim(train_x)[[1]] ,
    "\tNo of test samples\t",dim(test_x)[[1]])


#Defining the model
#a linear stack of layers
model<-keras_model_sequential()

model %>%
  #defining a 2-D convolution layer
  layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",
                input_shape=c(32,32,3)
                ) %>%
  layer_activation("relu") %>%
  #another 2-D conv layer
  layer_conv_2d(filter=32 ,kernel_size=c(3,3))  %>%
  layer_activation("relu") %>%
#Defining a Pooling layer which reduces the dimentions of the features map and reduces the 
#computational complexity of the model
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25) %>% #dropout layer to avoid overfitting 
  
  
  layer_conv_2d(filter=32 , kernel_size=c(3,3),padding="same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter=32,kernel_size=c(3,3) ) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  #flatten the input
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  #output layer-10 classes-10 units
  layer_dense(10) %>%
  #applying softmax nonlinear activation function to the output layer to calculate
  #cross-entropy
  layer_activation("softmax") #for computing Probabilities of classes-"logit(log probabilities)

  
  
  
  
  



