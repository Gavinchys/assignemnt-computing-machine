#Removing all objects from memory.
rm(list=ls())

#Loading required packages.
library(tidyverse)
library(keras)
library(reticulate)
library(ggplot2)
library(cluster)
library(ClusterR)
library(readxl)
library(magrittr)
library(dplyr)

################################Part 3##########################################
#Importing data into R as a data frame.
mydata.3 = read_xlsx("automobile.xlsx", col_names = T)

###Question (a)
#Calculating the correlation between the columns using only complete cases.
corr.matrix = cor(mydata.3, use = "complete.obs")
#Setting the diagonal of the matrix to NA since it always equals to 1.
diag(corr.matrix) = NA; round(corr.matrix, 4)
#Selecting the columns that has the highest correlation .
most.corr = which(abs(corr.matrix) == max(abs(corr.matrix), na.rm = T), arr.ind = T)
#Displaying the variables that are the most correlated with the correlation value.
most.corr; corr.matrix[1,4]

####Question (b)
#Excluding claimCount and claimCost
mydata.3.filtered = dplyr::select(mydata.3, !claimCount & !claimCost)

##Normal Q-Q plots of untransformed data excluding claimCount and claimCost 
temp.1 = colnames(mydata.3.filtered)
par(mfrow = c(1,length(temp.1)))
for(i in 1:length(temp.1)){
  var.1 = mydata.3.filtered %>% dplyr::select(temp.1[i]) %>% pull
  qqnorm(var.1, main = temp.1[i]); qqline(var.1, col = 2, lty = 2, lwd = 2)
}

#Log transformation of data
mydata.3.trans = mydata.3.filtered %>% mutate(carVal.log = log(carVal + 1))
#Removing untransformed carVal column
mydata.3.trans = dplyr::select(mydata.3.trans, !carVal)
#Rearranging columns of the data frame
mydata.3.trans = mydata.3.trans[,c(4,1:3)]

#Normal Q-Q plots of transformed data
temp.2 = colnames(mydata.3.trans)
par(mfrow = c(1,length(temp.2)))
for(i in 1:length(temp.2)){
  var.2 = mydata.3.trans %>% dplyr::select(temp.2[i]) %>% pull
  qqnorm(var.2, main = temp.2[i]); qqline(var.2, col = 2, lty = 2, lwd = 2)
}

#PCA with cor = TRUE to scale and center the data
mydata.3.pca = princomp(mydata.3.trans, cor = TRUE)
summary(mydata.3.pca)
#Calclating the cumulative variance explained by the PC 
cum.var = cumsum(mydata.3.pca$sdev^2); total.var = sum(mydata.3.pca$sdev^2)
#Selecting least number of PC that retain 80% of variation in data
PC.retained = which((cum.var / total.var) >= 0.8)[1]; PC.retained

###Question (c)
##Skree Plot
#Preparing the data frame containing proportion of variance explained by individual PC and cumulative variance
temp.3 = data.frame(mydata.3.pca$sdev^2 / total.var, cum.prop.var = cum.var/total.var)

#Plotting the skree plot
ggplot(data = temp.3) +
  geom_path(mapping = aes(x = row.names(temp.3),y = temp.3[,1], color = "Individual"), group = 1) +
  geom_point(mapping = aes(x = row.names(temp.3),y = temp.3[,1], color = "Individual")) +
  geom_path(mapping = aes(x = row.names(temp.3), y = temp.3[,2], color = "Cumulative"),group = 1) + 
  geom_point(mapping = aes(x = row.names(temp.3), y = temp.3[,2], color = "Cumulative")) +
  #Adding a y-intercept at 0.8
  geom_hline(aes(yintercept = 0.8), linetype = 2) +
  labs(x = "Principal Component", y = "Proportion of Explained Variance") +
  guides() +
  theme(legend.position = "right", legend.title = element_blank()) + 
  #Specifying scale of the y axis
  scale_y_continuous(breaks  = seq(from = 0, to = 1, by = 0.2))

##Biplot
library(ggbiplot)
#Plotting a biplot between the 3 PC selected, alpha = 0.004 chosen to improve readability of plot
ggbiplot(mydata.3.pca, choices = c(1,2), alpha = 0.004, circle = T, ellipse = T, varname.size = 3)
ggbiplot(mydata.3.pca, choices = c(1,3), alpha = 0.004, circle = T, ellipse = T, varname.size = 3)
ggbiplot(mydata.3.pca, choices = c(2,3), alpha = 0.004, circle = T, ellipse = T, varname.size = 3) 


  
###Question (d)
#Preparing data frame consisting of the claimCost and scores of the first 3 PC
model.df = data.frame(mydata.3$claimCost, mydata.3.pca$scores[,-4])
colnames(model.df)[1] = "claimCost"
#Fitting the linear model with the first 3 PC
model.3 = lm(claimCost ~ Comp.1 + Comp.2 + Comp.3, data = model.df); summary(model.3)

###Question (e)
#New set of features
carVal.new = 22; carAge.new = 14.5; distance.new = 25; duration.new = 10
#Transforming data
carVal.new.log = log(carVal.new + 1)
#Preparing vector of data to be used for prediction
new.obs = c(duration.new, distance.new, carVal.new.log, carAge.new)
#Scaling and centering the data to be used for prediction
new.obs.standardized = (new.obs - mydata.3.pca$center)/mydata.3.pca$scale

#Preparing a matrix containing the loadings of the first 3 PC
PC = matrix(nrow = 4, ncol = 3)
for(i in 1:3){
  PC[,i] = mydata.3.pca$loadings[,i]
}

#Matrix multiplication between data used for prediction and loadings 
PC.new.obs = as.data.frame(new.obs.standardized %*% PC)
colnames(PC.new.obs) = colnames(model.df[,-1])
#Predicting the claimCost of policy with the specifications of the new set of features
prediction = predict(model.3, newdata = PC.new.obs); prediction



################################Part 4##########################################
#Importing data into R as a data frame.
mydata.4 = read.csv("freMTPL2freq.csv")

#Corrections made to the dataset
mydata.4$ClaimNb = pmin(mydata.4$ClaimNb, 4)
mydata.4$Exposure = pmin(mydata.4$Exposure, 1)

##Create factors for columns that has a character data type.
for(i in seq_along(mydata.4)){
  if(is.character(mydata.4[[i]])){
    mydata.4[[i]] <- factor(mydata.4[[i]])
  }
}

###Question (a)
str(mydata.4)

#Author of function: Dr George Tzougas
#Function obtained from Week 9 Lecture Slides Part VI B
#Min-Max scaler used to normalize continuous predictors between -1 and 1
MM_scaling = function(data){
  2 * (data - min(data))/(max(data) - min(data)) - 1
}

#Min-Max Scaling of continuous predictors
mydata.4.NN = data.frame(ClaimNb = mydata.4$ClaimNb)
mydata.4.NN$Area = MM_scaling(as.integer(mydata.4$Area))
mydata.4.NN$VehPower = MM_scaling(as.numeric(mydata.4$VehPower))
mydata.4.NN$VehAge = MM_scaling(as.numeric(mydata.4$VehAge))
mydata.4.NN$DrivAge = MM_scaling(mydata.4$DrivAge)
mydata.4.NN$BonusMalus = MM_scaling(mydata.4$BonusMalus)
mydata.4.NN$VehGas = MM_scaling(as.integer(mydata.4$VehGas))
mydata.4.NN$Density = MM_scaling(mydata.4$Density)

#Learning Testing Split
#Learning Sample Index
learn.idx = sample(1:nrow(mydata.4), round(0.9 * nrow(mydata.4)), replace = F)
#Learning Testing Split
learn = mydata.4[learn.idx,] ; test = mydata.4[-learn.idx,]
learn.NN = mydata.4.NN[learn.idx,] ; test.NN = mydata.4.NN[-learn.idx,]

#Embedding Layers
library(tensorflow)
#Number of vehicle brands and region 
Br_ndistinct = length(unique(learn$VehBrand)) 
Re_ndistinct = length(unique(learn$Region))

#Setting up input layer for categorical features
VehBrand = layer_input(shape = c(1), dtype = 'int32', name = 'VehBrand')
Region = layer_input(shape = (1), dtype = 'int32', name = 'Region')

#Dimension of embedding layer for VehBrand and Region
qEmb = 2

#Creating embedding layer for VehBrand and Region
BrEmb = VehBrand %>% 
  layer_embedding(input_dim = Br_ndistinct, output_dim = qEmb,
                  input_length = 1, name = "BrEMB") %>%
  layer_flatten(name = "Br_flat")
ReEmb = Region %>%
  layer_embedding(input_dim = Re_ndistinct, output_dim = qEmb, 
                  input_length = 1, name = 'ReEmb') %>%
  layer_flatten(name = 'Re_flat')


###Question (b)
#Set design matrix for continuous variables
Design.learn = as.matrix(learn.NN[, -1]); Design.test = as.matrix(test.NN[, -1])

#Set matrices for categorical variables
Br.learn = as.matrix(as.integer(learn$VehBrand)) - 1; Br.test = as.matrix(as.integer(test$VehBrand)) - 1
Re.learn = as.matrix(as.integer(learn$Region)) - 1; Re.test = as.matrix(as.integer(test$Region)) - 1 

#Non-trainable offset 
Exp.learn = as.matrix(learn$Exposure); Exp.test = as.matrix(test$Exposure)
Exp.log.learn = log(Exp.learn); Exp.log.test = log(Exp.test)

#Matrix for response variable
Y.learn = as.matrix(learn.NN$ClaimNb); Y.test = as.matrix(test.NN$ClaimNb)

##Neural Network
#Setting hyperparameters
#Number of neurons in each layer
q1 = 25; q2 = 10; q3 = 15
#Number of epochs and number of samples per gradient upgrade
epochs = 1000; batchsize = 10000

#Setting up Input layer for continuous features
Design = layer_input(shape = ncol(learn.NN) - 1, dtype = 'float32', name = "Design")

#Setting up Input layer for log(Exposure) as offset 
Exp.log = layer_input(shape = c(1), dtype = 'float32', name = 'Exp.log')
Exp = layer_input(shape = c(1), dtype = 'float32', name = 'Exp')

#Main architecture with 3 hidden layers
Network = list(Design, BrEmb, ReEmb) %>% layer_concatenate(name = 'concate') %>%
  
  #1st hidden layer
  layer_dense(units = q1, activation = 'tanh', name = 'hidden1') %>%
  
  #2nd hidden layer
  layer_dense(units = q2, activation = 'tanh', name = 'hidden2') %>%
  
  #3rd hidden layer
  layer_dense(units = q3, activation = 'tanh', name = 'hidden3') %>%
  
  #output layer (w/ one neuron only)
  layer_dense(units = 1, activation = 'linear', name = 'Network')
  

#Output layer to combine main architecture and offset layer 
Response = list(Network, Exp.log) %>%
  
  #Adding exposure and the last neuron
  layer_add() %>%
  
  #Give the response
  layer_dense(units = 1,
              activation = 'exponential',
              name = 'Response',
              trainable = FALSE,
              weights = list(array(1, dim = c(1,1)), array(0,dim = c(1))))


#Assembling the model
model.4 = keras_model(inputs = c(Design, VehBrand, Region, Exp.log), 
                      outputs = c(Response))
summary(model.4)


#Configuring the model
model.4 %>% compile(
  #Poisson deviance loss
  loss = 'poisson',
  #Nadam optimizer
  optimizer = 'nadam'
)
#Model fitting by running gradient descent method to minimize obj. function
{
  t1 = proc.time()
  
  fit = model.4 %>% fit(
    list(Design.learn, Br.learn, Re.learn, Exp.log.learn), #Predictors
    Y.learn, #Response
    
    verbose = 1,
    
    epoch = epochs,
    
    batch_size = batchsize,
    
    validation_split = 0.2 #20% of learning dataset as validation set
  )
  print(proc.time() - t1)
}

#Predicted value of claim numbers
learn$nn0 = as.vector(model.4 %>% predict(list(Design.learn, Br.learn, Re.learn, Exp.log.learn)))
test$nn0 = as.vector(model.4 %>% predict(list(Design.test, Br.test, Re.test, Exp.log.test)))

#Author of function: Dr George Tzougas
#Function obtained from Week 8 Slides II Lecture Notes Part VI A
#Used to calculate the Poisson deviance loss
dev.loss = function(y, mu, density.func){
  logL.tilde = log(density.func(y, y))
  logL.hat = log(density.func(y, mu))
  2 * mean(logL.tilde - logL.hat)
}

#Deviance loss for the deep neural network on learning dataset
dev.loss(y = learn$ClaimNb, mu = learn$nn0, density.func = dpois)
#Deviance loss for the deep neural network on teset dataset
dev.loss(y = test$ClaimNb, mu = test$nn0, density.func = dpois)