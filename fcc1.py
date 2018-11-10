"""
Working with nltk and FCC data.

Using NLTK for corpus construction
This is a count based (tf-idf) approach using sklearn's TfIDF vectorizer
Dimensionality reduction using sklearn PC and SVD.
 
"""
#import os
import nltk
from nltk.corpus import PlaintextCorpusReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
#from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd

print("Creating corpus")

corpus_root = './docs/'		#directory path for txt files
#ls -l ./docs | wc -l #to get number of files in dir 
newCorpus = PlaintextCorpusReader(corpus_root, '[a-zA-Z0-9_\-]+.txt')

files = newCorpus.fileids()

print('Extracting text from all docs..')
docs=[] #empty list, append from corpus
row=[] #row index to grab hand coded labels and filer names from external csv file
for f in files:
	docs.append(newCorpus.raw(fileids=f))
	row.append(int(re.sub('(file)|(-[1-9].txt)','',f)))  #document numbers

## vectorize the words (i.e. count and transform in one step): unigrams and bigrams

print('Creating term document matrix with TfIdf vectorizer..')
vectorizer = TfidfVectorizer(min_df=0.15, max_df=.5, stop_words='english', ngram_range=(1,2), encoding='utf-8')
X = vectorizer.fit_transform(docs)

print("n_samples: %d, n_features: %d" % X.shape)


""" #to see the vocabulary or extract specific feature names.
print(vectorizer.vocabulary_) # not X.vocabulary_ 
feature_names = vectorizer.get_feature_names()
for feat in range(1,X.shape[1]):
	print(feature_names[feat]) #not X.feature_names[]
"""

## K-means clustering in the original space (before 2D reduction)
print('K-means clustering in original vector space.')
km = KMeans(n_clusters=5, init = 'k-means++', n_init=15, max_iter=100)
km.fit(X)	 

## Dimensionality reduction using SVD (as in Latent Semantic Analysis)
print('Truncated SVD to reduce dimensionality.')
svd = TruncatedSVD(n_components=2).fit(X)
X2D_svd = svd.transform(X)

## Clustering using k-means, separately for each reduced space.  
print('K-means clustering in reduced 2D SVD space..')
km_svd = KMeans(n_clusters=5, init = 'k-means++', n_init=15, max_iter=100)
km_svd.fit(X2D_svd)	

### Dimensionality reduction using PCA 
print('PCA to reduce dimensionality.')
Xdense = X.toarray() #PCA does not suppose sparse matrix

pca = PCA(n_components=2).fit(Xdense)  
X2D_pca = pca.transform(Xdense)


print('K-means clustering in reduced 2D PCA space..')
km_pca = KMeans(n_clusters=5, init = 'k-means++', n_init=15, max_iter=100)
km_pca.fit(X2D_pca)	

#Subplots
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, sharex=False, sharey=False)
ax1.scatter(X2D_svd[:,0], X2D_svd[:,1], c = km.labels_)
ax1.set_title('SVD space (KM-orig)')
ax2.scatter(X2D_pca[:,0], X2D_pca[:,1], c = km.labels_)
ax2.set_title('PCA space (KM-orig)')
ax3.scatter(X2D_svd[:,0], X2D_svd[:,1], c = km_svd.labels_)
ax3.set_title('SVD space (KM-2D)')
ax4.scatter(X2D_pca[:,0], X2D_pca[:,1], c = km_pca.labels_)
ax4.set_title('PCA space (KM-2D)')

plt.show(block=True)

combo = pd.concat([ pd.DataFrame(row,columns=['Rows']),\
					pd.DataFrame(X2D_svd[:,0],columns=['x_svd']),\
					pd.DataFrame(X2D_svd[:,1],columns=['y_svd']),\
					pd.DataFrame(X2D_pca[:,0],columns=['x_pca']),\
					pd.DataFrame(X2D_pca[:,1], columns=['y_pca']),\
					pd.DataFrame(km.labels_, columns=['lab_km']), \
					pd.DataFrame(km_svd.labels_, columns=['lab_km_svd']),\
					pd.DataFrame(km_pca.labels_, columns=['lab_km_pca'])],\
					axis=1)
#print(combo.loc[1:3,:])

#Saving to csv to ship out to D3 plotting, and other analyses e.g. classification.
#row = row.insert(0,0)
gp_info = pd.read_csv('./FCC_grouped.csv', header=0,usecols=[0,2,3])#, skiprows=lambda x: x not in row)
#print(gp_info.loc[1:3,['idx','filer','gp']]), # or specific col as gp_info.gp

merged_info = pd.merge(gp_info,combo, how='inner',left_on='idx',right_on='Rows')
#print(merged_info.loc[1:10,:])

merged_info.to_csv("./pyFCCout.csv", index=False)

#quit()

