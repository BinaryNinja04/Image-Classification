## Loading Packages
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("EBImage")
install.packages("keras")
install.packages("tensorflow")

library(EBImage)
library(keras)

## Reading Images
setwd("C:/Users/Pranav Shrivastava/OneDrive/Pictures/classification")
pics <- c('p1.jpeg', 'p2.jpeg', 'p3.jpeg', 'p4.jpeg', 'p5.jpeg',
          'c1.jpeg', 'c2.jpeg', 'c3.jpeg', 'c4.jpeg', 'c5.jpeg')

mypic <- list()
for(i in 1:10){mypic[[i]] <- readImage(pics[i])}

##Explore
print(mypic[[1]])       ## displays dimensions of the picture
display(mypic[[8]])     ## displaying pic in viewer
summary(mypic[[8]])
hist(mypic[[2]])
str(mypic)

## Resizing all images to have exactly the same dimensions
for(i in 1:10) {mypic[[i]] <- resize(mypic[[i]], 28, 28)}

## Reshape
for(i in 1:10) {mypic[[i]] <- array_reshape(mypic[[i]], c(28, 28, 3))}

## Row Bind ( we still had 10 different items in the list, so we will need to combine these)
trainx <- NULL
for (i in 1:4){trainx <- rbind(trainx, mypic[[i]])}
str(trainx)

for (i in 6:9){trainx <- rbind(trainx, mypic[[i]])}
str(trainx)

textx <- rbind(mypic[[5]], mypic[[10]])
trainy <- c(0,0,0,0,1,1,1,1)      # 0 -> plane 1-> car (response variable on the basis of trainx) 
testy <- c(0,1)                   # 0 -> plane 1-> car (response variable on the basis of testx)

## One Hot Encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

##Model
model <- keras_model_sequential()
model %>% 
  layer_dense(units=256, activation = 'relu', input_shape = c(2352)) %>% 
  layer_dense(units=128, activation = 'relu') %>% 
  layer_dense(units=2, activation = 'softmax')
summary(model)

## Compiling
model %>% 
  compile(loss = 'binary_crossentropy',     #as our response variable has only 2 values
          optimizer = optimizer_rmsprop(),
          metrics = c('accuracy'))

#Fit Model
history <- model %>% 
  fit(trainx,
      trainLabels,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.2)
plot(history)

##Evaluation & Prediction -> training data
model %>% evaluate(trainx, trainLabels)
pred <- model %>% predict_classes(trainx)
table(Predicted = pred, Actual = trainy)
prob <- model %>% predict_proba(trainx)
cbind(prob, Predicted = pred, Actual = trainy)

##Evaluation & Prediction -> test data
model %>% evaluate(testx, testLabels)
pred <- model %>% predict_classes(testx)
table(Predicted = pred, Actual = testy)
prob <- model %>% predict_proba(testx)
cbind(prob, Predicted = pred, Actual = testy)