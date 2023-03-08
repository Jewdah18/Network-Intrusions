# Importing Packages

library(cluster)
library(clustMixType)

# Loading the Data

# set the correct file Path
setwd("C:/Users/jdrel/OneDrive/Documents/Data_Science/Springboard/Capstone-2")

# Load the data
data <- read.csv("./data/interim/X_scaled.csv")
clusters <- read.csv("./data/interim/clusters.csv")

# Create a loop for numbers 36-42
for (i in 36:42) {
    # Reassign those columns as factors
    data[i] <- lapply(data[i], factor)
}

#Initialize Silhouette Scores
sil_score <- numeric()
# Initialize dissimilarity matrix for each row
diss_mat <- numeric()

# Create the first loop that will loop through all the rows
for (i in seq_len(1)) {
    for (j in seq_len(nrow(data))) {
        # combine two rows at a time to make it small enough to handle
        rows <- rbind(data[1, ], data[j, ])
        # Create dissimilarity matrix to compare the points in the selected rows
        diss_mat_temp <- daisy(rows, metric = "gower",
                        type = list(as.integer = seq_len(35),
                                    as.character = 36:42))
        # Change dissmat to a list so it can be flattened
        diss_mat <- c(diss_mat, list(diss_mat_temp))
  }
    cat("The length of the dissimilarity matrix is:", diss_mat)
    # Get the label for that specific row
    label <- clusters$cluster[i]
    cat("and the cluster label is:", label)
    # Run the silhouette score for that row
    sil_score <- c(sil_score, silhouette(diss_mat, label)[, 3])
}

# Print out the silhouette score
print(mean(sil_score))