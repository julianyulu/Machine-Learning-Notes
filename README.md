# Machine-Learning-Notes
----  
Collection of my hand-written notes and lectures pdfs while taking *Coursea* course *Machine Learning* by **Andrew Ng**  
# Summary of contents  
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
In this section I'll summarize a few important points when applying machine learning in real coding precedure, such as the importance of standardize features in some situiation, as well as normalize samples in some other situations. These practical experience are from exercises on [DataCamp](http://datacamp.com/) and [Udacity](https://www.udacity.com/). More summaries will be added as the learning goes.  

## Improve performance of clustering (unsupervised learning)  
As known to all, clustering is a unsupervised learning method based on minimizing the total initia for clusters, with given the number of clusters. Importantly, one has to realize that there are two situations that could lead to poor performance by clustering method (e.g. KMeans) if not taken care of:
+ Case 1: When many features are not on the same scale, e.g. 10 people v.s. 1000000 dollars. 
+ Case 2: When there is large variance between samples while only the trends are of interest, e.g. stock price of different companies over the years.  

**Reason that the above two case matters**: the reason roots in "Larger variance makes bigger influence".  

**Solution to case 1**: to reduce the impact of features with large variance, standardize the feature. Each feature is normalized to their standard deviation after substracting their mean, e.g. *StandardScaler* in *sklearn.preprocessing*.  

**Solution to case 2**: this is not widely known, one needs to normalize samples. Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of other samples so that its norm (l1 or l2) equals one. Note that this works for samples, compare to *case 1* which works for features. e.g. *Normalizer* in *sklearn.preprocessing*.  

The above solution could lead to huge imporvement for clustering.  




# Creation  
2017.12.15 - 2018.05.05  
NOTABILITY Version 7.2 by &copy Ginger Labs, Inc.   

# Last update  
May. 05, 2018  

# Claim of rights   
All original lecture content and slids copy rights belongs to Andrew Ng, the lecture notes and and summarization are based on the lecture contents and free to use and distribute according to GPL.  
