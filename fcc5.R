#Reading and manipulating txt files from R to work with tidytxt, stringr etc
library(readr)
library(stringr)
library(magrittr)
library(tidytext)
library(tidyverse)

files<-dir(path="./docs/")
for (f in files){
	txt<-read_lines(file=paste0("./docs/", f, sep=""),) %>% str_c(collapse=1)   
	#txt<- str_c(txt,collapse=1)
	txt <- gsub("([.:;,§¶&]+)|([0-9]+)|([()]+)|([\f])+|([-–+/“”’]+)", " ", txt) 
	
	txt_df <- data_frame(line = 1:length(txt), text=txt)
	txt_df %>% unnest_tokens(word,text)

	ngm <- txt_df %>% unnest_tokens(bigram, text, token = "ngrams", n = 2)	

	txt <- gsub(" ", "_",ngm[,"bigram"])
	txt <- gsub("(\")"," ",txt)
	txt <- gsub("(,_)+|([()])","",txt)  
	txt%>% write_lines(,path=paste0("./tempdocs/bigram/",f, sep=""))	
}

