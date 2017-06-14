#Deep learning in R
#Installing Keras and Tensorflow at the Backend

#You also need to install Rtools for installing 'reticulate' package for
#using python in R 
#You can install Rtools for your R version here--
#https://cran.r-project.org/bin/windows/Rtools/


#The reticulate package provides an R interface to Python modules, classes, and 
#functions
install.packages('reticulate')
require(reticulate)


#installing 'devtools' package for installing Packages from github
install.packages('devtools')

devtools::install_github("rstudio/keras") 
#installing keras 
#It will first install tensorflow then keras.


 
#The above code will install the keras library from the GitHub repository.

#loading keras in R 
library(keras)
#The R interface to Keras uses TensorFlow as it’s underlying computation engine.
#So we need to install Tensorflow engine
install_tensorflow()

#For installing a gpu version of Tensorflow
install_tensorflow(gpu = T)


#Getting started with Keras for R
#The core data structure of Keras is a model, a way to organize layers. 
#The simplest type of model is the Sequential model, a linear stack of layers. 
#For more complex architectures, you should use the Keras functional API, 
#which allows to build arbitrary graphs of layers.

#loading the keras inbuilt mnist dataset
data<-dataset_mnist()
?dataset_mnist #MNIST database of handwritten digits


#Training Data
train_x<-data$train$x
train_y<-data$train$y

#Test Set
test_x<-data$test$x
test_y<-data$test$y

#converting a 2D array into a 1D array for feeding 
#into the MLP and normalising the matrix
train_x<-array(train_x,dim=c(dim(train_x)[1],prod(dim(train_x)[-1])))/255
test_x<-array(train_x,dim=c(dim(train_x)[1],prod(dim(train_x)[-1])))/255


#converting the target variable to once hot encoded 
#vectors using keras inbuilt function

train_y<-to_categorical(train_y,10)
test_y<-to_categorical(test_y,10)

#Now defining a keras MLP sequential model
model <- keras_model_sequential()

#defining the model with 1 input layer[784 neurons], 1 hidden layer[784 neurons] 
#with dropout rate 0.4 and 1 output layer[10 neurons]
#i.e number of digits from 0 to 9

model %>% 
  #Add a densely-connected NN layer to an output
  layer_dense(units=784,input_shape=784)  %>%
  #dropout layer to prevent Overfitting
  layer_dropout(rate=0.4) %>%
  #Apply an activation function to an output.
  #Relu can only be used for Hidden layers
  layer_activation(activation="relu") %>%
  layer_dense(units=10) %>%
  #Apply an activation function to an output layer
  #softmax activation for Output layer which computes the probabilities for the classes
  layer_activation(activation="softmax") 

