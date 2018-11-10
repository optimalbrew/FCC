"""
Working with fastText for classification of the FCC files

"""
import string
import re
import glob # for unix style file paths with some patterns https://docs.python.org/3.5/library/glob.html

import fastText
from fastText import load_model
from fastText import util

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns



#removing punctuation from strings courtesy SparkAndShine + ShadowRanger on
#https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python

print(string.punctuation)
#'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

#underscores are required by FastText for __label__, so get rid of that from list
punc = string.punctuation.replace('_','')

"""
string.translate(s, table[, deletechars])
Delete all characters from s that are in deletechars (if present), and then translate the 
characters using table, which must be a 256-character string giving the translation for 
each character value, indexed by its ordinal. If table is None, then only the character 
deletion step is performed.

string.maketrans(from, to)
Return a translation table suitable for passing to translate(), that will map each character 
in from into the character at the same position in to; from and to must have the same length.
"""
table = str.maketrans(dict.fromkeys(punc)) #from 

#source text files
#path2dir = './pymode/pyCorpus/*.txt'  #play directory
path2dir = './docs/*.txt'  #actual directory

print('Read files and prep for fastext')

#read the text files and write the contents of each into separate lines of a single file.
#create (or overwrite) 
with open(file='./tempdocs/corpus.txt', mode='w') as corpus:
	for doc in glob.glob(path2dir):
		with open(doc,'r', newline=None) as f:
			fdata = f.read().replace('\n',' ').lower().translate(table) #
			fdata = re.sub(r'[0-9:]','',fdata)			#remove numbers (not okay in all contexts) 
			print(fdata,file = corpus)

#estimated training time on all 377 doc corpus is ETA about 2 hours on my machine!
#so limit it to a subset for testing..  take a 5% random sample of the docs which still takes 20 minutes

#cat ./tempdocs/corpus.txt | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .05) print $0}' > ./tempdocs/corpus5.txt
#should keep track of which ones were used.

f=fastText.FastText.train_unsupervised('./tempdocs/corpus5.txt',
	 lr=0.1, dim=100, ws=5, epoch=5, minCount=1, minCountLabel=0, minn=0, maxn=0, neg=5,
	  wordNgrams=1, loss='softmax', bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001,
	   label='__label__', verbose=2, pretrainedVectors='')
	
#save the model for future use
f.save_model('saved_fastTextModel')

#reload model later without having to train again
f = load_model('saved_fastTextModel') 

#Info about filers, needed for labeling, index refers to row in the master csv
gp = pd.read_csv('pyFCCout.csv', usecols=[0,1,2])
gp.head()
#    idx                                              filer        gp
# 0    3       NCTA - The Internet & Television Association     Cable
# 1    4                                Comcast Corporation     Cable
# 2    5       NCTA - The Internet & Television Association     Cable
# 3    6                                AT&T Services, Inc.  Carriers
# 4    7  American Electric Power Service Corporation,So...     Power
gp.columns = ['Row','Filer','Group']
set(gp['Group'])
#{'others', 'Muni', 'Cable', 'Power', 'Carriers', 'FCC'}

gp['Group'].describe()
# count       377
# unique        6
# top       Cable
# freq        137
# Name: Group, dtype: object


#use the trained model to create document vectors
with open(file='./tempdocs/doc2Vec.csv',mode='w') as docvec:
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
docVectors = pd.read_csv('tempdocs/doc2Vec.csv',index_col = 0, header=None) 

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


#Subplots using seaborn

# Set up the matplotlib figure
f, axes = plt.subplots(2,2, figsize=(8, 8))
#sns.despine(left=True)
sns.scatterplot(x="x_svd", y="y_svd", hue="Group", data=merged, ax = axes[0,0])
axes[0,0].set_title('SVD Space')
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('')
sns.scatterplot(x="x_tSNE10", y="y_tSNE10", hue="Group", data=merged, ax = axes[0,1])
axes[0,1].set_title('t-SNE Space (perplexity=10)')
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('')
sns.scatterplot(x="x_tSNE20", y="y_tSNE20", hue="Group", data=merged, ax = axes[1,0])
axes[1,0].set_title('t-SNE Space (perplexity=20)')
axes[1,0].set_xlabel('')
axes[1,0].set_ylabel('')
sns.scatterplot(x="x_tSNE30", y="y_tSNE30", hue="Group", data=merged, ax = axes[1,1])
axes[1,1].set_title('t-SNE Space (perplexity=30)')
axes[1,1].set_xlabel('')
axes[1,1].set_ylabel('')
#plt.show()
plt.savefig('images/fastCommonLang.png')







#getting nearest neighbors: just a basic similarity ranking. Can be used for doc neighbors
######################################
# def find_nearest_neighbor(query, vectors, ban_set, cossims=None):
#     """
#     query is a 1d numpy array corresponding to the vector to which you want to
#     find the closest vector
#     vectors is a 2d numpy array corresponding to the vectors you want to consider
#     ban_set is a set of indicies within vectors you want to ignore for nearest match
#     cossims is a 1d numpy array of size len(vectors), which can be passed for efficiency
#     returns the index of the closest match to query within vectors
#     """
#     if cossims is None:
#         cossims = np.matmul(vectors, query, out=cossims)
#     else:
#         np.matmul(vectors, query, out=cossims)
#     rank = len(cossims) - 1
#     result_i = np.argpartition(cossims, rank)[rank]
#     while result_i in ban_set:
#         rank -= 1
#         result_i = np.argpartition(cossims, rank)[rank]
# return result_i
########################################

# Retrieve list of normalized word vectors for the first words up
# until the threshold count.

# Gets words with associated frequency sorted by default in descending order
words, freq = f.get_words(include_freq=True) #freq not used
#words = words[:args.threshold]
words = words[:500]
	
vectors = np.zeros((len(words), f.get_dimension()), dtype=float)
for i in range(len(words)):
	wv = f.get_word_vector(words[i])
	wv = wv / np.linalg.norm(wv)
	vectors[i] = wv

query = f.get_sentence_vector('secret lies within')

nn1 = util.find_nearest_neighbor(
		query, vectors, ban_set=set(), cossims=None
	)
#print(nn1)
words[nn1]




