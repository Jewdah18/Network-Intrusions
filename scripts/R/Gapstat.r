# Importing Packages

library(cluster)
library(clustMixType)
library(kproto)

# Loading the Data

# set the correct file Path
setwd("C:/Users/jdrel/OneDrive/Documents/Data_Science/Springboard/Capstone-2")

# Load the data
data <- read.csv("./data/interim/Cluster_features.csv")

# Convert the dummy variables into categorical variables for the metric
df[, 5:9] <- lapply(df[, 5:9], as.factor)

dissim <- daisy(data, metric = "gower")

k <- 8

gap_stat <- clusGap(data = data, FUN = function(k) {
    kproto(data, k = k, cat.vars(c(5:9)))
}, B = 50, diss = dissim, K.crit = Null)

print(gap_stat)