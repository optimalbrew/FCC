"""
Train models on AWS

The corpus files must already be there from S3
fastText must be installed.

On a t-2 medium, training on the 400 doc corpus takes 8 hours!
Switched to t2 extra large, and also used a 20% sample.

If using pre-trained model, use US-west-1 not oregon (west-2). 

Do the dimensionality reduction (SVD/tSNE) on AWS as well and save the csv to S3.

"""
# import string
# import re
# import glob # for unix style file paths with some patterns https://docs.python.org/3.5/library/glob.html

import fastText
from fastText import load_model
from fastText import util

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

gp = pd.read_csv('pyFCCout.csv', usecols=[0,1,2])

#for corpus in corpusList:

#corpus = 'corpus' #can loop through a list as well.. 
corpusList = ['corpus', 'cableCorpus', 'powerCorpus', 'carrierCorpus']

for corpus in corpusList
	f=fastText.FastText.train_unsupervised('./tempdocs/'+corpus+'.txt',
		 lr=0.1, dim=100, ws=5, epoch=5, minCount=1, minCountLabel=0, minn=0, maxn=0, neg=5,
		  wordNgrams=1, loss='softmax', bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001,
		   label='__label__', verbose=2, pretrainedVectors='')

	#save the model for future use
	f.save_model('saved_'+corpus+'.model')

	#reload model later without having to train again
	#f = load_model('saved_fastTextModelCable') 


	#use the trained model to create document vectors
	with open(file='./tempdocs/doc2VecCable5.csv',mode='w') as docvec:
		with open('./tempdocs/corpus.txt', 'r', newline='\n') as d:
			for idx,line in enumerate(d):
				#replace newlines and gibberish from pdf as needed. Could be done earlier with punctuation
				line=line.replace('\n'," ").replace('“'," ").replace('”'," ").replace('\x0c'," ").\
							replace('¶',"").replace('\\n'," ").replace('§§'," ")
				sv = f.get_sentence_vector(line)
				dvStr = str(list(sv)).replace('[',"").replace(']',"")
				print(str(idx)+','+dvStr,file=docvec) #0-indexed csv for pandas
				#can be read in as data = pd.read_csv('tempdocs/doc2Vec.csv',index_col = 0)
				#wc -l ./tempdocs/doc2vec.csv to check the size

	#read in the vectors for dimensionality reduction (SVD) / or 2D embedding (tSNE)
	docVectors = pd.read_csv('tempdocs/doc2VecCable5.csv',index_col = 0, header=None) 

	# from sklearn.decomposition import TruncatedSVD
	svd = TruncatedSVD(n_components=2).fit(docVectors)
	docVec2D_svd = svd.transform(docVectors)
	docVec2D_svd.shape
	#t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE
	#from sklearn.manifold import TSNE
	##ranges: perplexity 5-50 (def 30) related to # of neighbors, lr 10-1000 (def 200)

	#Alternative values of perplexity
	docVec_embedded10 = TSNE(n_components=2, perplexity=10.0, learning_rate=200.0).fit_transform(docVectors)
	docVec_embedded20 = TSNE(n_components=2, perplexity=20.0, learning_rate=200.0).fit_transform(docVectors)
	docVec_embedded30 = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0).fit_transform(docVectors)
	#docVec_embedded.shape


	#Merge dataframes together
	merged = pd.concat([gp,
			pd.DataFrame(docVec2D_svd[:,0:2],columns=['x_svd','y_svd']),
			pd.DataFrame(docVec_embedded10[:,0:2],columns=['x_tSNE10','y_tSNE10']),
			pd.DataFrame(docVec_embedded20[:,0:2],columns=['x_tSNE20','y_tSNE20']),
			pd.DataFrame(docVec_embedded30[:,0:2],columns=['x_tSNE30','y_tSNE30'])],
			axis=1)

	merged.to_csv('fit'+corpus+'.csv', index = False)


#Subplots using seaborn
#done offline not on AWS

import matplotlib.pyplot as plt
import seaborn as sns


fitList = ['fitcorpus10', 'fitcorpus20', 'fitpowerCorpus', 'fitcarrierCorpus']

for fit in fitList:
	merged = pd.read_csv('awsFit/'+fit+'.csv')
	# Set up the matplotlib figure
	f, axes = plt.subplots(2,2, figsize=(8, 8))
	#sns.despine(left=True)
	sns.scatterplot(x="x_svd", y="y_svd", hue="gp", data=merged, ax = axes[0,0])
	axes[0,0].set_title('SVD Space')
	axes[0,0].set_xlabel('')
	axes[0,0].set_ylabel('')
	sns.scatterplot(x="x_tSNE10", y="y_tSNE10", hue="gp", data=merged, ax = axes[0,1])
	axes[0,1].set_title('t-SNE Space (perplexity=10)')
	axes[0,1].set_xlabel('')
	axes[0,1].set_ylabel('')
	sns.scatterplot(x="x_tSNE20", y="y_tSNE20", hue="gp", data=merged, ax = axes[1,0])
	axes[1,0].set_title('t-SNE Space (perplexity=20)')
	axes[1,0].set_xlabel('')
	axes[1,0].set_ylabel('')
	sns.scatterplot(x="x_tSNE30", y="y_tSNE30", hue="gp", data=merged, ax = axes[1,1])
	axes[1,1].set_title('t-SNE Space (perplexity=30)')
	axes[1,1].set_xlabel('')
	axes[1,1].set_ylabel('')
	#plt.show()
	plt.savefig('images/'+fit+'.png')




