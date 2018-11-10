# Visualization of filings on FCC policy proposals
This project uses simple natural language processing tools such as FastText for word2vec embeddings, and count-based methods such as Tf-Idf to visualize the differnet positions taken by distinct stakeholders in the communication policy space. The visualization is based on standard dimensionality reduction techniques (singular value decomposition and t-SNE).

One issue that the FCC recently addressed was about restrictions placed by local governments on access to utility poles which may be owned by local governments or utility companies. Poles not only carry power lines, but also support communication network gear such as fiber optic cables and related hardware. The poles thus have a *power space* (transmission lines, usually higher up the pole) and also a *communication space* (usually lower on the pole). Network gear may be attached directly to the poles (pole mount), or attached to steel cables running along poles (strand mounts). There are regional variations on how quickly and easily communication companies can obtain permissions to access poles to attach new equipment or modify existing equipment. Naturally different parties have startkly different views and these are represented in filings by individuals and organizations (including lobbying groups).  

## Common text corpus 
Visualizing document embeddings in 2D space SVD and t-SNE (with varying perplexitities) when the embeddings are created from a corpus containing sample filings from all groups. 
![common curpus](./fastCommonLang.png)


## Cable and ISP company text corpus 
Visualizing document embeddings created from a corpus containing sample filings from only cable and internet service providers. 
![cable corpus](./fastCableLang.png)

In this initial attempt, the filings by different (hand labeled) groups such as utility companies, cable companies, local government, wireless carriers do not exhibit any clear clustering. This may be because the corpora used to create the embeddings were too small (5 percent random subsamples used, to save on training time). 
