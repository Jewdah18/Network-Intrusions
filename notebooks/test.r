# Importing Packages

library(cluster)
library(clustMixType)

# Loading the Data

# set the correct file Path
setwd("C:/Users/jdrel/OneDrive/Documents/Data_Science/Springboard/Capstone-2")

# Load the data
data <- read.csv("./data/interim/X_scaled.csv")
clusters <- read.csv("./data/interim/clusters.csv")

# Separate out numerical and categorical
data[, (ncol(data) - 6):ncol(data)] <- lapply(data[, (ncol(data) - 6):ncol(data)], factor)

# Create dissimilarity matrix to compare all the different points
dissimilarity_matrix <- daisy(data, metric = "gower")

print(head(dissimilarity_matrix))

#import the clusters from EDA
clus <- clusters$cluster

# Compute the silhouette score with a dissimilarity matrix that uses Gower
# distance so that both numerical and categorical variables are taken into
# account
silhouette_coefficient <- silhouette(clus, dissimilarity_matrix)
mean_silhouette_coefficient <- mean(silhouette_coefficient[, 3])