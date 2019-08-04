########Module 4 Mini project #################
##########Personel loan campaign ########
#######Saurabh Mudgal 24th March 2019####

setwd("C:/BACP/Module 4 - Machine learning/Project/")
loan_cart=read.table("PL_XSELL.csv",sep = ",",header = T)
View(loan_cart)
table(loan_cart$TARGET)

names(loan_cart)
str(loan_cart)
View(loan_cart)

#converting acc_op_date to date format
loan_cart$ACC_OP_DATE = as.Date(loan_cart$ACC_OP_DATE, format="%m/%d/%Y")
str(loan_cart)
                                
#split data in test and train data set 
set.seed(123)  ## to get reapeatable data 

ind  = sample(2,nrow(loan_cart),replace = TRUE ,prob = c(.7,.3))
dev_cart = loan_cart[ind==1,]
hold_out_cart = loan_cart[ind==2,]  
table(dev_cart$TARGET)
table(hold_out_cart$TARGET)

names(dev_cart)
hold_out_cart = loan_cart[ind==2,]  
nrow(dev_cart)
nrow(hold_out_cart)

#lib for CART
library(rpart)
#lib for visual flow chart of tree 
library(rpart.plot)

#setting up control parameter 
r.ctrl = rpart.control(minsplit = 100,minbucket = 10,cp=0,xval=10)

#build tree
m_cart=rpart(formula=TARGET ~.,data = dev_cart[,-1], method = "class", control = r.ctrl)
m_cart
rpart.plot(m_cart)



#how the tree performs 
printcp(m_cart)
plotcp(m_cart)

##prun tree

## As per rpart results , use cp = .0019
p_m_cart = prune(m_cart, cp=.0019,"cp")
printcp(p_m_cart)
rpart.plot(p_m_cart)
printcp(p_m_cart)

##Model evaluation

dev_cart$predict.class = predict(p_m_cart,dev_cart, type="class")
dev_cart$predict.score = predict(p_m_cart,dev_cart, type="prob")

View(dev_cart)
head(dev_cart)


## deciling code
decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (
    ifelse(x<deciles[1], 1,
           ifelse(x<deciles[2], 2,
                  ifelse(x<deciles[3], 3,
                         ifelse(x<deciles[4], 4,
                                ifelse(x<deciles[5], 5,
                                       ifelse(x<deciles[6], 6,
                                              ifelse(x<deciles[7], 7,
                                                     ifelse(x<deciles[8], 8,
                                                            ifelse(x<deciles[9], 9, 10
                                                            ))))))))))
}

class(dev_cart$predict.score)
## deciling
dev_cart$deciles <- decile(dev_cart$predict.score[,2])

View(dev_cart)
## Ranking code
install.packages("data.table")
install.packages("scales")
library(data.table)
library(scales)
tmp_DT_cart = data.table(dev_cart)
rank <- tmp_DT_cart[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round(rank$cnt_resp / rank$cnt,4);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),4);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),4);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp) * 100;
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)

install.packages("ROCR")
install.packages("ineq")
library(ROCR)
library(ineq)
pred <- prediction(dev_cart$predict.score[,2],dev_cart$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)

gini = ineq(dev_cart$predict.score[,2], type="Gini")

##classification error, confusion matrix
with(dev_cart, table(TARGET, predict.class))
auc
KS
gini

##      predict.class
#TARGET     0     1
#0 12103   210
#1  1358   436
#> auc
#[1] 0.7728194
#> KS
#[1] 0.3965317 - this is close to 40% hence its good model
#> gini
#[1] 0.4762494
#> (210+1358)/14107
#[1] 0.1111505
#
#11% is classification error 


View(rank)

##scoring test sample on hold out data#########

hold_out_cart$predict.class <- predict(p_m_cart, hold_out_cart, type="class")
hold_out_cart$predict.score <- predict(p_m_cart, hold_out_cart, type="prob")


hold_out_cart$deciles <- decile(hold_out_cart$predict.score[,2])


pred <- prediction(hold_out_cart$predict.score[,2],hold_out_cart$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
tmp_DT_cart = data.table(hold_out_cart)
h_rank <- tmp_DT_cart[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
h_rank$rrate <- round(h_rank$cnt_resp / h_rank$cnt,4);
h_rank$cum_resp <- cumsum(h_rank$cnt_resp)
h_rank$cum_non_resp <- cumsum(h_rank$cnt_non_resp)
h_rank$cum_rel_resp <- round(h_rank$cum_resp / sum(h_rank$cnt_resp),4);
h_rank$cum_rel_non_resp <- round(h_rank$cum_non_resp / sum(h_rank$cnt_non_resp),4);
h_rank$ks <- abs(h_rank$cum_rel_resp - h_rank$cum_rel_non_resp)*100;
h_rank$rrate <- percent(h_rank$rrate)
h_rank$cum_rel_resp <- percent(h_rank$cum_rel_resp)
h_rank$cum_rel_non_resp <- percent(h_rank$cum_rel_non_resp)


View(h_rank)

install.packages("ROCR")
install.packages("ineq")
library(ROCR)
library(ineq)
pred <- prediction(hold_out_cart$predict.score[,2],hold_out_cart$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)

gini = ineq(hold_out_cart$predict.score[,2], type="Gini")

##classification error, confusion matrix
with(hold_out_cart, table(TARGET, predict.class))
auc
KS
gini

####
#predict.class
#TARGET    0    1
#0 5056  119
#1  580  138
#> auc
#[1] 0.7728194
#> KS
#[1] 0.3965317
#> gini
#[1] 0.4762494
#> (119+580)/5893
#[1] 0.1186153
##classification error is 11% which is same as was in dev sample 
####

view(h_rank)


### model is not overfitted ###########




#######Random forest techniques####################

loan_ranForest=read.table("PL_XSELL.csv",sep = ",",header = T)
View(loan_ranForest)
#converting acc_op_date to date format
loan_ranForest$ACC_OP_DATE = as.Date(loan_ranForest$ACC_OP_DATE, format="%m/%d/%Y")


#split data in test and train data set 

ind  = sample(2,nrow(loan_ranForest),replace = TRUE ,prob = c(.7,.3))
dev_ranForest = loan_ranForest[ind==1,]
test_ranForest = loan_ranForest[ind==2,]
c(nrow(dev_ranForest),nrow(test_ranForest))
str(loan_ranForest)

#table(loan_ranForest$OCCUPATION)



library(randomForest)

## Calling syntax to build the Random Forest
RF <- randomForest(as.factor(TARGET) ~ ., data = dev_ranForest[,-1], 
                   ntree=501, mtry = 28,replace=FALSE, nodesize = 10,
                   importance=TRUE)
print(RF)

#> print(RF)

#Call:
#  randomForest(formula = as.factor(TARGET) ~ ., data = loan_ranForest[,      -1], ntree = 501, mtry = 10, nodesize = 10, importance = TRUE) 
#Type of random forest: classification
#Number of trees: 501
#No. of variables tried at each split: 10#

#OOB estimate of  error rate: 4.31%
#Confusion matrix:
 # 0    1  class.error
#0 17476   12 0.0006861848
#1   850 1662 0.3383757962

##lets identify optimum value of ntree and mtry

plot(RF, main="")

RF$err.rate##OOB stablizes around 40-70 tree

legend("topright", c("OOB", "0", "1"), text.col=1:6, lty=1:3, col=1:3)
title(main="Error Rates Random Forest RFDF")


## List the importance of the variables.
impVar <- round(randomForest::importance(RF), 2)
impVar[order(impVar[,3], decreasing=TRUE),]

##key varaiables as per meandecreaseGIni are balance,SCR,Acc_opp_Date

dev_ranForest[,-c(1,2)]
str(dev_ranForest)

str(dev_ranForest$ACC_OP_DATE)
## Tuning Random Forest
tRF <- tuneRF(x = dev_ranForest[,-c(1,2)], 
              y=as.factor(dev_ranForest$TARGET),
              mtryStart = 6, ##sqrt of predictor variables
              ntreeTry=55,  ##OOB Stablizes (should be odd number)
              stepFactor = 1.5, 
              improve = 0.0001, 
              trace = TRUE, 
              plot = TRUE,
              doBest = TRUE,
              nodesize = 10,  ###need to change for overfitting prob
              importance=TRUE
)

#mtry = 6  OOB error = 6.25% 
#Searching left ...
#mtry = 4 	OOB error = 6.9% 
#-0.1043084 1e-04 
#Searching right ...
#mtry = 9 	OOB error = 6.09% 
#0.0260771 1e-04 
#mtry = 13 	OOB error = 5.71% 
#0.06169965 1e-04 
#mtry = 19 	OOB error = 5.47% 
#0.04342432 1e-04 
#mtry = 28 	OOB error = 5.54% 
#-0.01426719 1e-04 

#least OOB error is 5.47% hence 19 variables are optimum



tRF$importance


##measure model performane 


dev_ranForest$predict.class = predict(tRF,dev_ranForest, type="class")
dev_ranForest$predict.score = predict(tRF,dev_ranForest, type="prob")

head(dev_ranForest)
class(dev_ranForest$predict.score)

## deciling
## deciling code
decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (
    ifelse(x<deciles[1], 1,
    ifelse(x<deciles[2], 2,
    ifelse(x<deciles[3], 3,
    ifelse(x<deciles[4], 4,
    ifelse(x<deciles[5], 5,
    ifelse(x<deciles[6], 6,
    ifelse(x<deciles[7], 7,
    ifelse(x<deciles[8], 8,
    ifelse(x<deciles[9], 9, 10
    ))))))))))
}


dev_ranForest$deciles <- decile(dev_ranForest$predict.score[,2])


library(data.table)
tmp_DT = data.table(dev_ranForest)
rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);

library(scales)
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)

sum(dev_ranForest$TARGET) / nrow(dev_ranForest)


library(ROCR)
pred <- prediction(dev_ranForest$predict.score[,2], dev_ranForest$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
KS

## Area Under Curve
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc

## Gini Coefficient
library(ineq)
gini = ineq(dev_ranForest$predict.score[,2], type="Gini")
gini

## Classification Error
with(dev_ranForest, table(TARGET, predict.class))


#RGET     0     1
#0 12313     0
#1   197  1597
##> (0+197)/14107
#[1] 0.0139647


## Scoring syntax


test_ranForest$predict.class <- predict(tRF, test_ranForest, type="class")
test_ranForest$predict.score <- predict(tRF, test_ranForest, type="prob")

test_ranForest$deciles <- decile(test_ranForest$predict.score[,2])

tmp_DT = data.table(test_ranForest)
h_rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
h_rank$rrate <- round (h_rank$cnt_resp / h_rank$cnt,2);
h_rank$cum_resp <- cumsum(h_rank$cnt_resp)
h_rank$cum_non_resp <- cumsum(h_rank$cnt_non_resp)
h_rank$cum_rel_resp <- round(h_rank$cum_resp / sum(h_rank$cnt_resp),2);
h_rank$cum_rel_non_resp <- round(h_rank$cum_non_resp / sum(h_rank$cnt_non_resp),2);
h_rank$ks <- abs(h_rank$cum_rel_resp - h_rank$cum_rel_non_resp);


library(scales)
h_rank$rrate <- percent(h_rank$rrate)
h_rank$cum_rel_resp <- percent(h_rank$cum_rel_resp)
h_rank$cum_rel_non_resp <- percent(h_rank$cum_rel_non_resp)

View(h_rank)


library(ROCR)
pred <- prediction(test_ranForest$predict.score[,2], test_ranForest$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
KS

## Area Under Curve
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc

## Gini Coefficient
library(ineq)
gini = ineq(test_ranForest$predict.score[,2], type="Gini")
gini

## Classification Error
with(test_ranForest, table(TARGET, predict.class))

#predict.class
#TARGET    0    1
#0 5172    3
#1  302  416
#> (3+302)/5893
#[1] 0.05175632

##Model is overfitting ################
#need to check with changing node size 10, 25#


####Neural Network#######

loan_NN =read.table("PL_XSELL.csv",sep = ",",header = T)
View(loan_NN)
table(loan_NN$TARGET)

names(loan_NN)
#convert factor to date 
#loan_NN$ACC_OP_DATE=  as.Date(loan_ranForest$ACC_OP_DATE,factor="%m%d%y")

#split data in test and train data set 
#set.seed(123)
ind  = sample(2,nrow(loan_NN),replace = TRUE ,prob = c(.7,.3))
dev_NN = loan_NN[ind==1,]
hold_out_NN = loan_NN[ind==2,]  

#names(dev_cart)
str(dev_NN)



##as we can't use categorical variables in NN - convert to integer

occ.matrix <- model.matrix(~ GENDER - 1, data = dev_NN)
dev_NN <- data.frame(dev_NN, occ.matrix)


occ.matrix <- model.matrix(~ OCCUPATION - 1, data = dev_NN)
dev_NN <- data.frame(dev_NN, occ.matrix)

cc.matrix <- model.matrix(~ AGE_BKT - 1, data = dev_NN)
dev_NN <- data.frame(dev_NN, occ.matrix)
                     
cc.matrix <- model.matrix(~ ACC_TYPE - 1, data = dev_NN)
dev_NN <- data.frame(dev_NN, occ.matrix)



str(dev_NN)

##remove categorical data from df##
dev_NN_new=within(dev_NN,rm(CUST_ID,ACC_OP_DATE,GENDER,OCCUPATION,AGE_BKT,ACC_TYPE))
str(dev_NN_new)

allvars = colnames(dev_NN_new)
predictorVars = allvars[!allvars%in%"TARGET"]
predictorVars = paste(predictorVars,collapse = "+")
form=as.formula(paste("TARGET~",predictorVars,collapse = "+"))
form
library(neuralnet)

NN_Model=neuralnet(formula=form,
                   data=dev_NN_new,
                   hidden = 2,
                   err.fct = "sse", 
                   linear.output = FALSE,
                   lifesign = "full",
                   lifesign.step=10,
                   threshold = .01,
                   stepmax = 2000)


plot(NN_Model)

dev_NN_new$prob = NN_Model$net.result[[1]]
View(dev_NN_new)

## The distribution of the estimated probabilities
quantile(dev_NN_new$prob, c(0,1,5,10,25,50,75,90,95,98,99,100)/100)
hist(dev_NN_new$prob)


## deciling code
decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (
    ifelse(x<deciles[1], 1,
           ifelse(x<deciles[2], 2,
                  ifelse(x<deciles[3], 3,
                         ifelse(x<deciles[4], 4,
                                ifelse(x<deciles[5], 5,
                                       ifelse(x<deciles[6], 6,
                                              ifelse(x<deciles[7], 7,
                                                     ifelse(x<deciles[8], 8,
                                                            ifelse(x<deciles[9], 9, 10
                                                            ))))))))))
}

## deciling
dev_NN_new$deciles <- decile(dev_NN_new$prob)


## Ranking code
##install.packages("data.table")
library(data.table)
library(scales)

tmp_DT = data.table(dev_NN_new)
rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)

##scale the data 


loan_NN1 =read.table("PL_XSELL.csv",sep = ",",header = T)
ind  = sample(2,nrow(loan_NN1),replace = TRUE ,prob = c(.7,.3))
dev_NN1 = loan_NN1[ind==1,]
hold_out_NN1 = loan_NN1[ind==2,]

View(dev_NN1)

occ.matrix <- model.matrix(~ GENDER - 1, data = dev_NN1)
dev_NN1 <- data.frame(dev_NN1, occ.matrix)


occ.matrix <- model.matrix(~ OCCUPATION - 1, data = dev_NN1)
dev_NN1 <- data.frame(dev_NN1, occ.matrix)

cc.matrix <- model.matrix(~ AGE_BKT - 1, data = dev_NN1)
dev_NN1 <- data.frame(dev_NN1, occ.matrix)

cc.matrix <- model.matrix(~ ACC_TYPE - 1, data = dev_NN1)
dev_NN1 <- data.frame(dev_NN1, occ.matrix)



dev_NN1_new=within(dev_NN1,rm(TARGET,CUST_ID,ACC_OP_DATE,GENDER,OCCUPATION,AGE_BKT,ACC_TYPE))
str(dev_NN1_new)
View(dev_NN1_new)


allvars = colnames(dev_NN1_new)
predictorVars = allvars[!allvars%in%"TARGET"]
predictorVars = paste(predictorVars,collapse = "+")
form1=as.formula(paste("TARGET~",predictorVars,collapse = "+"))

form1

dev_NN_Scaled = scale(dev_NN1_new)

dev_NN_Scaled <- cbind(dev_NN1[2], dev_NN_Scaled)
View(dev_NN_Scaled)

NN_Model_Scaled =neuralnet(formula=form1,
                   data=dev_NN_Scaled,
                   hidden = 2,
                   err.fct = "sse", 
                   linear.output = FALSE,
                   lifesign = "full",
                   lifesign.step=1,
                   threshold = .1,
                   stepmax = 2000)

plot(NN_Model_Scaled)



dev_NN_Scaled$prob = NN_Model_Scaled$net.result[[1]]
View(dev_NN_Scaled)

## The distribution of the estimated probabilities
quantile(dev_NN_Scaled$prob, c(0,1,5,10,25,50,75,90,95,98,99,100)/100)
hist(dev_NN_Scaled$prob)

dev_NN_Scaled$deciles = decile(dev_NN_Scaled$prob)


##model performance 

tmp_DT = data.table(dev_NN_Scaled)
rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)

## Assgining 0 / 1 class based on certain threshold
dev_NN_Scaled$Class = ifelse(dev_NN_Scaled$prob>0.40,1,0)
with( dev_NN_Scaled, table(TARGET, as.factor(Class)  ))

## We can use the confusionMatrix function of the caret package 
##install.packages("caret")
library(caret)
confusionMatrix(nn.dev$Target, nn.dev$Class)


## Error Computation
sum((nn.dev$Target - nn.dev$Prob)^2)/2

## Other Model Performance Measures

library(ROCR)
pred <- prediction(dev_NN_Scaled$prob, dev_NN_Scaled$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)

library(ineq)
gini = ineq(dev_NN_Scaled$prob, type="Gini")


auc
KS
gini



###check for the performance of model in test data set 

occ.matrix <- model.matrix(~ GENDER - 1, data = hold_out_NN)
hold_out_NN <- data.frame(hold_out_NN, occ.matrix)


occ.matrix <- model.matrix(~ OCCUPATION - 1, data = hold_out_NN)
hold_out_NN <- data.frame(hold_out_NN, occ.matrix)

cc.matrix <- model.matrix(~ AGE_BKT - 1, data = hold_out_NN)
hold_out_NN <- data.frame(hold_out_NN, occ.matrix)

cc.matrix <- model.matrix(~ ACC_TYPE - 1, data = hold_out_NN)
hold_out_NN <- data.frame(hold_out_NN, occ.matrix)

View(hold_out_NN)

hold_out_NN_new=within(hold_out_NN,rm(TARGET,CUST_ID,ACC_OP_DATE,GENDER,OCCUPATION,AGE_BKT,ACC_TYPE))
#str(dev_NN1_new)
#View(dev_NN1_new)


allvars = colnames(hold_out_NN_new)
predictorVars = allvars[!allvars%in%"TARGET"]
predictorVars = paste(predictorVars,collapse = "+")
form1=as.formula(paste("TARGET~",predictorVars,collapse = "+"))

form1

hold_out_NN_scaled = scale(hold_out_NN_new)
hold_out_NN_scaled <- cbind(hold_out_NN[2], hold_out_NN_scaled)
View(hold_out_NN_scaled)


compute.output = compute(NN_Model_Scaled, hold_out_NN_scaled)
compute.output
hold_out_NN_new$Predict.score = compute.output$net.result
View(hold_out_NN_new)


## deciling
hold_out_NN_new$deciles = decile(hold_out_NN_new$Predict.score)

#quantile(hold_out_NN_new$prob, c(0,1,5,10,25,50,75,90,95,98,99,100)/100)
#hist(dev_NN_Scaled$prob)

dev_NN_Scaled$deciles = decile(dev_NN_Scaled$prob)
###model performance 

tmp_DT = data.table(hold_out_NN_new)
h_rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
h_rank$rrate <- round (h_rank$cnt_resp / h_rank$cnt,2);
h_rank$cum_resp <- cumsum(h_rank$cnt_resp)
h_rank$cum_non_resp <- cumsum(h_rank$cnt_non_resp)
h_rank$cum_rel_resp <- round(h_rank$cum_resp / sum(h_rank$cnt_resp),2);
h_rank$cum_rel_non_resp <- round(h_rank$cum_non_resp / sum(h_rank$cnt_non_resp),2);
h_rank$ks <- abs(h_rank$cum_rel_resp - h_rank$cum_rel_non_resp);

library(scales)
h_rank$rrate <- percent(h_rank$rrate)
h_rank$cum_rel_resp <- percent(h_rank$cum_rel_resp)
h_rank$cum_rel_non_resp <- percent(h_rank$cum_rel_non_resp)

View(h_rank)
View(rank)


-----------------------------------------------------
  
  ##ensemble modeling 
  install.packages("caret", dependencies = c("Depends", "Suggests"))
  library(RANN)
  fitControl <- trainControl(
    method = "cv",
    number = 10,
    savePredictions = 'final',
    classProbs = T)
trainContr  
  



