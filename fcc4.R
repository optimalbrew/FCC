#FCC part 3, using the text mining package to get going. 

library(tm) #for "text mining" in R
#fcc <- VCorpus(DirSource("./docs/",encoding="UTF-8"), readerControl=list(language="en"))
fcc <- VCorpus(DirSource("./tempdocs/bigram/",encoding="UTF-8"), readerControl=list(language="en"))


#cleanup: stemming
# fcc<-tm_map(fcc, stemDocument)
# fcc <- tm_map(fcc, removeNumbers) #ASCII
# fcc <- tm_map(fcc, removeNumbers, ucp=T) #unicode
# fcc <- tm_map(fcc,stripWhitespace)
# fcc <- tm_map(fcc,removePunctuation, ucp=T)
# fcc <- tm_map(fcc,removePunctuation, ucp=F)
# fcc <- tm_map(fcc, removeWords, stopwords("en")) #inspect(fcc) will reveal a reduction in char count

del_words= c("commission", "section","counsel", "dear","dortch","washington","and_the","by_the","dc_re","dear_ms","docket_no","dortch_on", "dortch_secretary","ex_parte","for_the","h_dortch","marlene_h",       
"ms_dortch","no_dear","notice_of","of_a", "of_the","on_the","secretary_federal","street_sw","th_street","that_the","the_above","the_commission",        
"to_the","washington_dc","wc_docket","with_the")  
#fcc <- tm_map(fcc, removeWords, del_words)
# Term Doc Matrices and Document Term 



tdm_fcc <- TermDocumentMatrix(fcc, control = list(
												weighting = function(x) weightTfIdf(x, normalize = T),
												tokenize = "Boost",
												tolower = T,
												removePunctuation = F,
												removeNumbers = T,
												stopwords = c(stopwords("en"),del_words),
												stemming = F))

dim(tdm_fcc) # 16000, 20 in current example
#to see a few
#tdm_fcc$dimnames$Terms[10:50]
#inspect(tdm_fcc)


#function to get the hand coded group (row in data file) from document name
stripTxt <- function(x){
	#x<-gsub('[a-zA-Z]+',"",x)
	x<-gsub('(file)|(-[0-9].txt)','',x)
}

#Grab the previously (manually) added group tags by extracting row numbers (stripTxt)
r1<-as.numeric(lapply(Docs(tdm_fcc),stripTxt))
#then use these as row indexes to read the filer names or group from FCC data file.
fccFile<-read.csv("FCC_grouped.csv", header=T)
gp<-fccFile[r1,4] #group tags
filers <- fccFile[r1,3] #filer names

#Idx <- sample(length(r1),10)  a smaller sample for testing 
rm(fccFile)

library(ggplot2)
library(grid)
#For Jupyter plot sizing
#library(repr) #representaion of strings and bytes from R to other apps
#options(repr.plot.width=4, repr.plot.height=3) #default is 7x7 



#sparsity levels and numplots
# numr <-3 ; numc <- 3
# pushViewport(viewport(layout = grid.layout(numr,numc)))
# spar_level = matrix(c(.9,.8,.7,.6,.5,.4,.3,.2,.1),numr,numc,byrow=T) #sparsity level

# numr <-2 ; numc <- 2
# pushViewport(viewport(layout = grid.layout(numr,numc)))
# spar_level = matrix(c(.6,.5,.4,.3),numr,numc,byrow=T) #sparsity level

numr <-1 ; numc <- 1
pushViewport(viewport(layout = grid.layout(numr,numc)))
spar_level = matrix(c(.5),numr,numc,byrow=T) #sparsity level



for(i in 1:numr){
	for(j in 1:numc){
			
			tdm_fcc_tmp <- removeSparseTerms(tdm_fcc, sparse = spar_level[i,j]) 
			#cat(paste("\nTerms for sparsity level ",as.character(spar_level[i,j]),"\n" ))
			#cat(tdm_fcc_tmp$dimnames$Terms)
			
			x<- svd(tdm_fcc_tmp,2,2) #or svd(A,2,2) to restrict to 2 singular values
			sigma <- x$d 
			S <- x$u #notation from tutorial is S and U not U and Vt
			U <-x$v
			
			#Restrict sigma to 2 singular values
			sigma2<-sigma[1:2]
			S2<-c(S[,1], S[,2])
			dim(S2)<-c(dim(tdm_fcc_tmp)[1],2)
			#if we had V-Transpose instead, (U here), then would have taken top two rows.
			U2<-c(U[,1], U[,2])
			dim(U2)<-c(dim(tdm_fcc_tmp)[2],2)
			U2<-t(U2)
			
			#Now the terms and documents
			sigma2Matrix<-c(sigma2[1], 0, 0, sigma2[2])
			dim(sigma2Matrix)<-c(2,2)
			
			#Scaled terms-concept matrix # uncomment when interested in plotting the terms in the  vector space
			#S2Scaled<-S2 %*% sigma2Matrix
			
			#scaled concept-document matrix: to see documents in vector space
			U2tScaled<-sigma2Matrix %*% U2
		
			#plotting
			layer_point1 <- geom_point(mapping = aes(x=U2tScaled[1,Idx], y=U2tScaled[2,Idx], color = gp[Idx]),size=1)#, shape=gp[Idx])
 			p1 <- ggplot() + layer_point1 + labs(x= paste("SVD with sparsity: ",as.character(spar_level[i,j]),", ", as.character(length(tdm_fcc_tmp$dimnames$Terms)),					" Terms"), 
                                y="", 
                                color="") +
     		theme(panel.grid.major.y = element_blank()) +
	    	 	theme(legend.position="top")
			
			print(p1, vp = viewport(layout.pos.row = i , layout.pos.col = j ))
	}
}

# Using principal components for visualization.

tdm_main <- removeSparseTerms(tdm_fcc, sparse = .8)
tdm_main$dimnames$Terms
 
#note transpose (DTM not TDM)
z<-prcomp(t(tdm_main),rank=2,scale=T) #obtain the first 2 principal components

#repeated from above
numr <-2 ; numc <- 2
pushViewport(viewport(layout = grid.layout(numr,numc)))
spar_level = matrix(c(.5),numr,numc,byrow=T) #sparsity level

## Using K-means for grouping (and this is obviously in the original space, not the 2D space)
cl <- kmeans(t(tdm_main), centers=6, nstart=25) #again note transpose (DTM not TDM)


#Using SVD for visualization
 
x<- svd(tdm_main,2,2) #restrict to 2 singular values
sigma <- x$d 
S <- x$u 
U <-x$v
			
sigma2<-sigma[1:2] #not neccessary with above restriction
S2<-c(S[,1], S[,2])
dim(S2)<-c(dim(tdm_main)[1],2)
#if we had V-Transpose instead, (U here), then would have taken top two rows.
U2<-c(U[,1], U[,2])
dim(U2)<-c(dim(tdm_main)[2],2)
U2<-t(U2)

#Now the terms and documents
sigma2Matrix<-c(sigma2[1], 0, 0, sigma2[2])
dim(sigma2Matrix)<-c(2,2)

#Scaled terms-concept matrix # uncomment when interested in plotting the terms in the  vector space
#S2Scaled<-S2 %*% sigma2Matrix

#scaled concept-document matrix: to see documents in vector space
U2tScaled<-sigma2Matrix %*% U2

#plotting
layer_point1 <- geom_point(mapping = aes(x=U2tScaled[1,], y=U2tScaled[2,], color = gp),size=1)
p1 <- ggplot() + layer_point1 + labs(x= paste("SVD and hand coded groups"), 
                                y="", 
                                color="") +
     		theme(panel.grid.major.y = element_blank()) +
    	 	theme(legend.position="top")

layer_point1 <- geom_point(mapping = aes(x=U2tScaled[1,], y=U2tScaled[2,], color = as.factor(cl$cluster)),size=1)
p2 <- ggplot() + layer_point1 + labs(x= paste("SVD and K-means clustering"), 
                                y="", 
                                color="") +
     		theme(panel.grid.major.y = element_blank()) +
    	 	theme(legend.position="top")


layer_point1 <- geom_point(mapping = aes(x=z$x[,1],y=z$x[,2], color = gp),size=1)#, 
p3 <- ggplot() + layer_point1 + labs(x= paste("Princ. Comp. and hand coded groups"), 
                                 y="", 
                                 color="") +
      		theme(panel.grid.major.y = element_blank()) +
     	 	theme(legend.position="top")


layer_point1 <- geom_point(mapping = aes(x=z$x[,1],y=z$x[,2], color = as.factor(cl$cluster)),size=1)#, 
p4 <- ggplot() + layer_point1 + labs(x= paste("Princ. Comp. and K-means cluster."), 
                                 y="", 
                                 color="") +
      		theme(panel.grid.major.y = element_blank()) +
     	 	theme(legend.position="top")

print(p1, vp = viewport(layout.pos.row = 1 , layout.pos.col = 1 ))
print(p2, vp = viewport(layout.pos.row = 1 , layout.pos.col = 2 ))
print(p3, vp = viewport(layout.pos.row = 2 , layout.pos.col = 1 ))
print(p4, vp = viewport(layout.pos.row = 2 , layout.pos.col = 2 ))


newData <- cbind(as.character(filers), as.character(gp), as.numeric(cl$cluster),z$x[,1],z$x[,2], U2tScaled[1,],U2tScaled[2,])
colnames(newData) <- c("filers","group","cluster","PC1", "PC2", "SV1", "SV2")
#write.csv(newData,file="fcc_vizData.csv",row.names=F)
write.csv(newData,file="fcc_vizDatabigram.csv",row.names=F)