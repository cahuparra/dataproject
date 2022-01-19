## ----setup, include=FALSE-----------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----dataset_upload,  message=FALSE, warning=FALSE----------------------------------------------------------------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")



library(tidyverse)
library(caret)
library(rpart)
library(data.table)
library(lubridate)
library(readxl)
library(e1071)
library(RColorBrewer)
library(rpart.plot)
library(matrixStats)

# Reviewing if file is loaded in current directory
path<-getwd()
file<-"WA_Fn-UseC_-HR-Employee-Attrition.csv"
filename<-paste(path,"/",file,sep="")
if (file.exists(filename)) {
  cat("Reading Attrition File")
} else {
  cat("Attrition file does not exist\n")
  cat("Please load file: ", file,"at directory: ",path,"\n")
  stop("Attrition file not found")
}
#load Attrition file
data_attrition<-read.csv("./data/WA_Fn-UseC_-HR-Employee-Attrition.csv")



## ----dataset_tyding,  message=FALSE, warning=FALSE----------------------------------------------------------------------------------------------------------------
# Convert to categorical, eliminate not relevant variables 
set.seed(7, sample.kind = "Rounding") 
data_clean<-data_attrition %>%
  mutate(Attrition<-ifelse(Attrition=="Yes",1,0),
         BusinessTravel<-as.factor(BusinessTravel),
         Department<-as.factor(Department),
         EducationField<-as.factor(EducationField),
         Gender<-as.factor(Gender), 
         JobRole<-as.factor(JobRole), 
         MaritalStatus<-as.factor(MaritalStatus), 
         OverTime<-as.factor(OverTime)) %>%
    select(EmployeeNumber, Attrition,  BusinessTravel, DailyRate, Department, 
           DistanceFromHome, Education, EducationField, EnvironmentSatisfaction, 
           Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, 
           JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, 
           NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, 
           RelationshipSatisfaction, StandardHours, StockOptionLevel, 
           TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, 
           YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, 
           YearsWithCurrManager)

data_clean$Age<-data_attrition[,1] # Fix Age title Attrition file
# Convert categoric to numeric values
data_numeric<-data_clean
data_numeric$Attrition<-as.numeric(as.factor(data_numeric$Attrition))
data_numeric$BusinessTravel<-as.numeric(as.factor(data_numeric$BusinessTravel))
data_numeric$Department<-as.numeric(as.factor(data_numeric$Department))
data_numeric$EducationField<-as.numeric(as.factor(data_numeric$EducationField))
data_numeric$Gender<-as.numeric(as.factor(data_numeric$Gender))
data_numeric$JobRole<-as.numeric(as.factor(data_numeric$JobRole))
data_numeric$MaritalStatus<-as.numeric(as.factor(data_numeric$MaritalStatus))
data_numeric$OverTime<-as.numeric(as.factor(data_numeric$OverTime))
# Generate dataset  for train function section 2.3 (caret package)
attrition_x<- data_numeric[,-2]
attrition_y<-data_numeric[,2]
# Scales rows of the dataset
x_centered <- sweep(attrition_x, 2, colMeans(attrition_x))
x_scaled <- sweep(x_centered, 2, colSds(as.matrix(attrition_x)), FUN = "/")
# Partition dataset into test and training section 2.3
test_index <- createDataPartition(attrition_y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- data_clean$Attrition[test_index]
train_x <- x_scaled[-test_index,]
# Eliminate NaaN column
train_x <- train_x[-23]
train_y <- data_clean$Attrition[-test_index]


## ----dataset_split,  message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------------------------
test_index<- createDataPartition(data_clean$Attrition,times = 1, p= 0.5, list=FALSE)
test_set<-data_clean %>% slice(test_index)
train_set<-data_clean %>% slice(-test_index)
test_index<- createDataPartition(data_numeric$Attrition,times = 1, p= 0.5, list=FALSE)
testnum_set<-data_numeric %>% slice(test_index)
trainnum_set<-data_numeric %>% slice(-test_index)


## ----tree_method,  message=FALSE, warning=FALSE-------------------------------------------------------------------------------------------------------------------

#generate the tree
train_rpart<-rpart(Attrition~., data=trainnum_set)
rpart_preds <- predict(train_rpart, testnum_set)
rpart_preds_fact<-ifelse(rpart_preds<1.5,1,2)
cc<-confusionMatrix(factor(rpart_preds_fact), factor(testnum_set$Attrition))

#Plot the tree
{rpart.plot(train_rpart)
title(main = "Decision Tree Attrition")}
# Calculate Accruracy, Sensitivity and Specificity
cc<-confusionMatrix(factor(rpart_preds_fact), factor(testnum_set$Attrition))
attr_results <- tibble(Method = "Regression Tree:", Accuracy = cc$overall["Accuracy"], Sensitivity = cc$byClass[1], Specificity = cc$byClass[2], Balanced_Accuracy = cc$byClass[11] ) 
attr_results %>% knitr::kable()


## ----tree_pruned,  message=FALSE, warning=FALSE-------------------------------------------------------------------------------------------------------------------
# Pruning Branches
train_rpart <- train(Attrition ~ ., method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.1, 0.002)), data = trainnum_set)  
rpart_preds <- predict(train_rpart, testnum_set)
# train_rpart <- train(Attrition ~ ., method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)), data = #train_set)  
# rpart_preds <- predict(train_rpart, test_set)
rpart_preds_fact<-ifelse(rpart_preds<1.5,1,2)
cc<-confusionMatrix(factor(rpart_preds_fact), factor(trainnum_set$Attrition))
# Plot tree
{rpart.plot(train_rpart$finalModel)
title(main = "Pruned Decision Tree Attrition")}

#Estimate Sensitivity and Specificity (Pruned Branches)
attr_results <- bind_rows(attr_results,tibble(Method = "Pruned Tree:", Accuracy = cc$overall["Accuracy"], Sensitivity = cc$byClass[1], Specificity = cc$byClass[2], Balanced_Accuracy = cc$byClass[11] )) 
attr_results %>% knitr::kable()


## ----Knn_PCA,  message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------------------------------
# Perform a principal component analysis of the scaled matrix.
# exclude Attrition from trainnum_set
Attrition<-trainnum_set$Attrition
pca <- prcomp(trainnum_set[-2])

#Plot the first two principal components with color representing PC1 and PC2.

data.frame(pc_1 = pca$x[,1], pc_2=pca$x[,2], Attrition=trainnum_set$At) %>%
  ggplot(aes(pc_1, pc_2, color=Attrition)) +
  geom_point() +
  ggtitle("Plot of Two Principal Component colored by Attrition")
# Boxplot of 4 first PCs grouped by Attrition
bpd <- data.frame(att = trainnum_set$Attrition, pca$x[,1:4]) %>%
  gather(key = "PC", value = "value", -att)
bpd %>% ggplot(aes(PC, value,fill= factor(att))) +
  geom_boxplot() + 
  ggtitle("Boxplot of four Principal Component grouped by Attrition",
          subtitle = "1- No Attrition, 2- Yes Attrition")
# HeatMap PC1 and PC2
sds <- matrixStats::colSds(as.matrix(trainnum_set)) 
ind <- order(sds, decreasing = TRUE)[1:50]
colors <- brewer.pal(7, "Dark2")[as.numeric(trainnum_set$Attrition)]
index<-c(ind[1], ind[2])
{heatmap(t(as.matrix(trainnum_set)[,index]), col = brewer.pal(11, "RdBu"), scale = "row", ColSideColors = colors, margins=c(6,6))
title("Heatmap Principal Components PC1 and PC2", cex.main = 2,)}
print("Relevance of firt two Principal Components: ") 
summary(pca)$importance[3,1:2]
index<-c(ind[1], ind[2]) #selecting PC1 and PC2
# Estimate conditional probability using PC1 and PC2
fit_knn <- knn3(trainnum_set[ ,index], factor(trainnum_set$Attrition),  k = 3)                   
y_hat_knn <- predict(fit_knn,
                     testnum_set[, index],
                     type="class")
#Estimate Sensitivity and Specificity (PCA)
cc<-confusionMatrix(y_hat_knn, factor(testnum_set$Attrition))
attr_results <- bind_rows(attr_results,tibble(Method = "Knn_PCA", Accuracy = cc$overall["Accuracy"], Sensitivity = cc$byClass[1], Specificity = cc$byClass[2], Balanced_Accuracy = cc$byClass[11] ))  
attr_results %>% knitr::kable()


## ----train_method_result,  message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------------------
# look up gamLoess
modelLookup("gamLoess") 
models<-c("glm", "gamLoess", "knn", "rf","qda","lda")
N<-6
fits <- lapply(models, function(model){ 
  print(model)
  train(train_x,train_y, method = model)
}) 

names(fits) <- models
pred <- sapply(fits, function(object){ 
  preds<-predict(object, newdata = test_x)
    return(preds)})

results <- sapply(c(1:N),function(i){ 
  cc<-confusionMatrix(factor(pred[,i]), factor(test_y)) 
  return(cc)
  })

for(i in 1:N){
  cc<-results[,i]
  attr_results <- bind_rows(attr_results,tibble(Method = models[i], Accuracy = cc$overall["Accuracy"], Sensitivity = cc$byClass[1], Specificity = cc$byClass[2], Balanced_Accuracy = cc$byClass[11] ))
}

attr_results %>% knitr::kable()




## ----result,  message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------------------------
# Display variables by importance

ggplot(varImp(fits[1]$glm)) + ggtitle("Variable Importance glm method")
ggplot(varImp(fits[3]$knn)) + ggtitle("Variable Importance knn method")
ggplot(varImp(fits[4]$rf)) + ggtitle("Variable Importance rf method")
ggplot(varImp(fits[5]$qda)) + ggtitle("Variable Importance qda method")
ggplot(varImp(fits[6]$lda)) + ggtitle("Variable Importance lda method")

# execute to generate R cmd file
# library(knitr)
# purl("Employee Attrition.Rmd", output = "employeeAttrition.R", documentation = 2)

