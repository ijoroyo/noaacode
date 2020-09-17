#- set working directory 
setwd("D:\\Documents\\Study and small Project training\\project_tsunami\\test\\noaa")
options(scipen = 999)

#- Load required libraries
library(caret)
library(nnet)
library(knitr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(ROCR)
library(bigmemory)
library(doParallel)
registerDoParallel(cores=8)

#- Import Data
data = read.csv("Gempa_Tsunami_NOAA_Bathy.csv", stringsAsFactors = F)

# - Univariate Analysis
# Tsunami and no tsunami in data set - Fair representation of both outcomes
kable(table(data$FLAG_TSUNAMI),
      col.names = c("Tsunami", "Frequency"), align = 'l')

# independent variable
ggplot(gather(data[,2:ncol(data)]), aes(value)) + 
  geom_histogram(bins = 5, fill = "blue", alpha = 0.6) + 
  facet_wrap(~key, scales = 'free_x')

str(data)

# Encode as a one hot vector multilabel data
#data <- cbind(data, class.ind(as.factor(data$TypeMag)))

# data-time type
#data$Date <- as.POSIXct(strptime(data$Date, "%m/%d/%Y"))

# check NA 
colSums(sapply(data, is.na))
#NA.Date <- data[is.na(data$Date), ]

#NA.Date$Date <- sapply(NA.Date$Time, function(x) unlist(strsplit(x, "T"))[1])
#NA.Date$Time <- sapply(NA.Date$Time, function(x) unlist(strsplit(x, "T"))[2])

#NA.Date$Date <- as.POSIXct(NA.Date$Date)
#NA.Date$Time <- sapply(NA.Date$Time, function(x) substr(x, 1, 8))

#data <- rbind(data[!is.na(data$Date), ], NA.Date)
#data <- data[order(data$Date), ]

# attach year/month/day as group variables 
# date.data <- sapply(as.character(data$Date), function(x) unlist(strsplit(x, "-")))
# 
# data$Year <- as.numeric(date.data[seq(1, 3*nrow(data), by = 3)])
# data$Month <- as.numeric(date.data[seq(2, 3*nrow(data), by = 3)])
# data$Day <- as.numeric(date.data[seq(3, 3*nrow(data), by = 3)])

# convert FLAG_TSUNAMI to factor
data$Actual = data$FLAG_TSUNAMI
data$FLAG_TSUNAMI = as.factor(data$FLAG_TSUNAMI)
levels(data$FLAG_TSUNAMI) = make.names(levels(factor(data$FLAG_TSUNAMI)))

# Visualiztion for tsunami 
data %>% 
  group_by(YEAR) %>% 
  summarise(Avg.num = n(), 
            Avg.mt = mean(FLAG_TSUNAMI, na.rm = T)) %>%
  ggplot(aes(x = YEAR, y = Avg.num)) + 
  geom_col(fill = "blue") + 
  stat_smooth(col = "red", method = "loess") + 
  labs(x = "Year",
       y = "Total Observations Tsunami Each Year",
       title = "Total Observations Tsunami Each Year (2008-2018)",
       caption = "Source: Significant Tsunami 2008-2018 by BMKG") + 
  theme_bw()

# Visualiztion for earthquake
data %>% 
  group_by(YEAR) %>% 
  summarise(Avg.num = n(), 
            Avg.mag = mean(EQ_PRIMARY, na.rm = T)) %>%
  ggplot(aes(x = YEAR, y = Avg.num)) + 
  geom_col(fill = "blue") + 
  stat_smooth(col = "red", method = "loess") + 
  labs(x = "Year",
       y = "Total Observations Earthquake Each Year",
       title = "Total Observations Earthquake Each Year (2008-2018)",
       caption = "Source: Significant Earthquake 2008-2018 by BMKG") + 
  theme_bw()

# Check out the average magnitude of all earthquakes happened each year.
data %>% 
  group_by(YEAR) %>% 
  summarise(Avg.num = n(), Avg.mag = mean(EQ_PRIMARY, na.rm = T)) %>%
  ggplot(aes(x = YEAR, y = Avg.mag)) + 
  geom_col(fill = "blue") + 
  labs(x = "Year",
       y = "Average Magnitude Each Year",
       title = "Total Observations Earthquake Each Year (2008-2018)",
       caption = "Source: Significant Earthquake 2008-2018 by BMKG") +  
  theme_bw()


# split data in training and test set.
#Index = sample(1:nrow(data), size = round(0.8*nrow(data)), replace=FALSE)
index <- c()
for (k in 1:length(data$YEAR)){
  if(data$YEAR[k] < 2020){
    index <- c(index, k)
  }
}

train = data[index ,]
test = data[-index ,]

rm(index)

#- set seed
set.seed(123)

#- Define controls
x = trainControl(method = "repeatedcv",
                 number = 10,
                 repeats = 3,
                 classProbs = TRUE,
                 summaryFunction = twoClassSummary)

#- train model
knn = train(FLAG_TSUNAMI~. , data = train[,2:15], method = "knn",
            preProcess = c("center","scale"),
            trControl = x,
            metric = "ROC",
            tuneLength = 10)

# print model results
knn

plot(knn)

test$Predicted = predict(knn, test[,2:15], "prob")[,2]

#- Area Under Curve
plot(performance(prediction(test$Predicted, test$Actual),
                 "tpr", "fpr"))

# use probability cut off 0.5 for classification
test$Predicted = ifelse(test$Predicted > 0.5, 1,0)

#- confusion matrix
confusionMatrix(factor(test$Predicted),
                factor(test$Actual))

# save the model to disk
saveRDS(knn, "./final_model_knn_predict.rds")

