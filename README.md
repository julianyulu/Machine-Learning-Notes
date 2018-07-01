# Machine-Learning-Notes
----  
Collection of my hand-written notes and lectures pdfs while taking *Coursea* course *Machine Learning* by **Andrew Ng**  
# Summary of contents  
----  
+ [Week1: Linear regression with one variable](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK1-Linear%20Regression%20With%20One%20Variable.pdf)  
  - Machine learning defination  
  - Supervised / Unsupervised Learning  
  - Linear regression with one variable
  - Cost function, learning rate
  - Batch gradient descent 

+ [Week2: Linear regression with multiple variables](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK2-Linear%20Regression%20with%20Multiple%20Variables%20.pdf)   
  - Multivariable linear regression  
  - Normal equation method  
  - Vectorization 
  
+ [Week3: Logistic Regressio and Regularization](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK3-Logistic%20Regression%20and%20Regularization%20.pdf)  
  - Classification 
  - Logistic regression: hypothesis representation, decision boundrary, cost function, gradient descent.
  - Optimization algorithms: Conjugate gradient, BFGS, L-BFGS
  - Multi-class classification: One-vs-All classification 
  - Overfitting: reduce feature space; regularization. 
  - Regularization and regularized linear/logistic regression, gradient descent
  
+ [Week4: Neural Networks: Representation](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK4-Neural%20Networks_Representation.pdf)  
  - Nonlinear hypothesis  
  - Motivation, representation   
  - Forward propagation algorithm  
  - Learning features: gate logic realization  

+ [Week5: Neural Networks: Learning](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK5-Neural%20Networks%20Learning%20.pdf)  
  - Classification problem  
  - Cost function  
  - Back propagation algorithm  
  - Gradient checking 
  - Random initialization: symmetry breaking 
  - Put it together  
  
+ [Week6: Evaluate a Learning Algorithm](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK6-Evaluate%20a%20Learning%20Algorithm%20.pdf)  
  - Decide what to try next 
  - Evaluate a hypothesis: training / testing data spliting 
  - Test error / Misclassification error 
  - Model selection: chosse right degree of polynomial 
  - Cross validation 
  - Diagnositic of bias VS variance
  - Bias-Variance trade off: choose regularization parameter 
  - Bias-Variance leanring curve 
  - Debugging a learning algorithm 
  - Neural networks and regularization 
  - Machine learning system design: recommandations and examples 
  - Error matrics for skewed classes: precission recall trade off, F score
  
+ [Week7: Support Vector Machines](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK7-Support%20Vector%20Machines.pdf)
  - Large margin classification
  - Cost function
  - Decision boundrary 
  - Kernels and similarity function, Mercer's Theroem
  - Linear kernel, gaussian kernel, poly kernel, chi-square kernel, etc
  - Choosing the landmarks
  - SVM parameters and multi-class classification 
  - Logistic regression VS SVMs

+ [Week8: Unsupervised Learning: Clustering, Dimentionality Reduction and PCA](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK8-Unsupervised%20Learning%20and%20PCA.pdf)
  - Clustering and application 
  - K-Means algorithm 
  - Optimization objective 
  - Random initialization 
  - Choose the number of clusters 
  - Dimentionality Reduction: data compression and visualization
  - Principal Componant Analysis: formulation, algorithm, reconstruction from compressed representation, advices
  
+ [Week9: Unsupervised Learning: Anormaly Detection and Recommender Systems](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK9-Anomaly%20Detection.pdf)
  - Density estimaiton 
  - Algorithm 
  - Building an anormaly detection system 
  - Anormaly detection VS supervised learning 
  - Chossing features: non-guassian feature transform, error analysis
  - Multi-variable gaussian distribution 
  - Recommender systems: content-based recommendations 
  - Recommender systems: Collaborative filtering 
  - Low-rank matrix factorization 
  - Recommender systems: mean normalization 

+ [Week10: Large Scale Machine Learning: Stochastic Gradient Descent and Online Learning](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK10-Large%20Scale%20Machine%20Learning%20.pdf)
  - STochastic gradient descent(SGD)
  - Mini-batch gradient descent 
  - SGD Convergence problem 
  - Online learning 
  - Map-reduce and data parallism 
  
+ [Week11: Application Example: Photo OCR](https://github.com/SuperYuLu/Machine-Learning-Notes/blob/master/HandWrittenNotes/WEEK11-Application%20Example%20Photo%20OCR.pdf)
  - Photo OCR piplines
  - Sliding window detection 
  - Artifical data synthesis 
  - Ceiling Analysis 



# Practical Notes  
----  
In this section I'll summarize a few important points when applying machine learning in real coding precedure, such as the importance of standardize features in some situiation, as well as normalize samples in some other situations. These practical experience are from exercises on [DataCamp](http://datacamp.com/), [Coursera](http://coursera.org/) and [Udacity](https://www.udacity.com/). More summaries will be added as the learning goes.  

## Importance of feature pre-processing and feature generation  
----  
Source: Coursera "How to win a data science competition: learn from to kagglers  
### Feature preprocessing 
+ Numeric feature 
  - Tree based method doesn't depend on scaling 
  - Non-tree based models hugely depend on scaling
  - Most often used preprocessing methods:
	- MinMaxScaler -> to [0, 1]
	- StandardScaler -> to 0 mean and 1 std 
	- Rand -> set spaces between sorted values to be equal
	- np.log(1 + x) or np.sqrt(1 + x) 
	
+ Categorical and ordinal feature
  -Oridinal feature: categorical features that are sorted in some meaningful order.  
  - One hot encoding: often used for non-tree based models
  - Label encoding: maps categories to numbers w/o extra numerical handling
  - Frequency encoding: maps categories to their appearing frequencies in the data 
  - Label and frequency encoding are ofen used for tree based 
  - Interation between categorical features: two individual categorical featureas A and B can be treated as a 2D feature, can help linear models and KNN
  
+ Datetime features 
  - Can generate features like: periodicity, time since, difference between date

+ Cooridinates  
  - Usually additional data will help
  - Can be used to generate new features like distances, raidus
  - Interesting to study centers of clusters
  - May consider rotated cooridinates or other reference frames
  
  
### Feature generation  
Feature generation is powered by:  
+ Prior knowledge 
+ Exploratory Data Analysis (EDA)  



## Improve performance of clustering (unsupervised learning)  
----  
As known to all, clustering is a unsupervised learning method based on minimizing the total initia for clusters, with given the number of clusters. Importantly, one has to realize that there are two situations that could lead to poor performance by clustering method (e.g. KMeans) if not taken care of:
+ Case 1: When many features are not on the same scale, e.g. 10 people v.s. 1000000 dollars. 
+ Case 2: When there is large variance between samples while only the trends are of interest, e.g. stock price of different companies over the years.  

**Reason that the above two case matters**: the reason roots in "Larger variance makes bigger influence".  

**Solution to case 1**: to reduce the impact of features with large variance, standardize the feature. Each feature is normalized to their standard deviation after substracting their mean, e.g. *StandardScaler* in *sklearn.preprocessing*.  

**Solution to case 2**: this is not widely known, one needs to normalize samples. Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of other samples so that its norm (l1 or l2) equals one. Note that this works for samples, compare to *case 1* which works for features. e.g. *Normalizer* in *sklearn.preprocessing*.  

The above solution could lead to huge imporvement for clustering.  

## Decide the number of clusters (unsupervised learning)  
----  
There are always the case when perforing clustering on samples but one doesn't know how many groups to cluster into, due to the nature of unsupervised learning. While there are two main methods that clustering can be perfomed, there are different ways to decide on the number of result clusters:  
+ KMeans: "elbow" on initia vs n_clusters plot, e.g. model.initia\_ after fit model using sklearn.cluster.KMeans. Where initia is "Sum of squared distances of samples to their closest cluster center" (sklearn.cluster.KMeans).
+ Hierarchical clustering: plot dendrogram and make decision on the maximum distance. e.g. use linkage, dendrogram and fcluster from scipy.cluster.hierarchical.  

## PCA: decide the number of principal components  
----  
When performing PCA for dimentionality reduction, one of the key steps is to make decision of the number of principal components. The underlie principle of PCA is that it rotates and shifts the feature space to find the principle axis which explains the maximal variance in data. Due to this feature, as similar to clustering, one has to take care of the variance in the feature space. Also note that PCA does not do feature selection as Lasso or tree model. The way to make decision on how many principal components is to make the bar plot of "explained variance" vs "pca feature", and choose the features that explains large portion of the variance. By doing this, one actually discovers the "intrinsic dimension of the data".  


## Visualizing high dimential data using t-SNE  
----  
High dimentional data is usually hard to visualize, expecially for unsupervised learning. PCA is on option, while another option t-SNE (t distrubuted stocastic neighbor embedding) can map high dementional data to 2D space while approximately preserves the nearness of data. e.g. using TSNE from sklearn.maniford.  


# Creation  
2017.12.15 - 2018.05.05  
NOTABILITY Version 7.2 by &copy Ginger Labs, Inc.   

# Last update  
July. 01, 2018  

# Claim of rights   
All original lecture content and slids copy rights belongs to Andrew Ng, the lecture notes and and summarization are based on the lecture contents and free to use and distribute according to GPL.  
