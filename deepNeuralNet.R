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

