#Deep learning in R
#Installing Keras and Tensorflow at the Backend

install.packages('devtools')

devtools::install_github("rstudio/keras") #installing keras requires latest version of  R

#updating R on Windows using installr package
#You also need to install Rtools33 or higher for installing keras library
install.packages("installr")

library(installr)
updateR() # will install the latest version of R and follow the instructions

 
#The above code will load the keras library from the GitHub repository.

#loading keras in R 
library(keras)
#The R interface to Keras uses TensorFlow as it’s underlying computation engine.
#So we need to install Tensorflow engine
install_tensorflow()
