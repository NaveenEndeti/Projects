

**K-Means Clustering Project 1**
**Problem Statement**:The main objective is to perform Modeling to the given data. We must try 5 different numbers of clusters based on the elbow curve and for each cluster visualize the clustering results.
**Solution**:We have uploaded a CSV file named Mall_customerrs to the Google Colab and read the given data in the file. K-Means is a distance-based algorithm, this difference of magnitude can create a problem. So let’s first bring all the variables to the same magnitude, we are performing Data Cleaning.  we are checking the basic data of null value or not in some entities like customer ID, annual income, age, and spending score. We Implemented K-Means the objective is to perform K-means clustering on the data and check the clustering metrics (inertia) Now, the objective of K-means is it take some random centroid and calculate the points attached to it and according to that centroid will be shifted. We will have a different cluster which depends on our initializers. Now We have initialized two clusters and pay attention – the initialization is not random here. If clusters are not initialized appropriately, K-Means can result in arbitrarily bad clusters. We have used the k++ initialization which generally produces better results. This is where K-Means++ helps.. Using the K-Means++ algorithm, we optimize the step where we randomly pick the cluster centroid. We are more likely to find a solution that is competitive with the optimal K-Means solution while using the K-Means++ initialization.

**Language Detection and Text Summarization Project 2**
**Problem Statement**:Most advances in NLP are geared toward English as a language. There are over 6000 living languages in the world, with the United Nations recognizing six official languages.
However, NLP techniques are primarily developed in the English language. The following are some of the barriers to NLP proliferation in non-English languages: The amount of research on NLP is heavily skewed toward the English language because it is the most commonly used language for business and academic communication worldwide. English is a high-resource language due to the availability of a large corpus of digital textual data, which is required for training and increasing the accuracy of NLP systems.
**Solution**:Model Architecture consists of an input dataset in 27 different languages, the preprocessing is done on the dataset, and after SVM (support vector machines) is used for the classification of
text. It is observed that the dataset works well for training the model and the model accuracy on the test data was about 85%.
Model=SVM
Train-Test-Split = 70:30
After training the model, the output of the predicted language text is passed to the translate method, it uses Google translator to translate the text in any language to English, and the translated text is used for summarization. In order to perform summarization we used cosine similarity between the word and summarized text with length reduced to the square root of the original text.
