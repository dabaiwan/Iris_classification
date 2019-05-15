#!/usr/bin/python
# -*- coding:utf8 -*-


#  Supervised learning example: Iris classification¶
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
iris = sns.load_dataset('iris')
iris.head()
sns.pairplot(iris, hue='species', height=1.5)

X_iris = iris.drop('species', axis=1)  # X_iris.shape is (150, 4)
y_iris = iris['species']    # y_iris.shape is (150, )

#   choose train_test_split
# from sklearn.cross_validation import train_test_split ; 这条语句报错，改为下面的语句
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)


from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data


from sklearn.metrics import accuracy_score
acc = accuracy_score(ytest, y_model)
print("预测精度是", acc)


# Unsupervised learning example: Iris dimensionality

from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)         # 4. Transform the data to two dimensions

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);


# Unsupervised learning: Iris clustering
# from sklearn.mixture import GMM    注： GMM已经停用，我们用GaussianMixture
from sklearn.mixture import GaussianMixture    # 1. Choose the model class
model = GaussianMixture(n_components=3,
            covariance_type='full')  # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)        # 4. Determine cluster labels

iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',
           col='cluster', fit_reg=False);
plt.show()