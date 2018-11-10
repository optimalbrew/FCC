#Exploratory FCC analysis part 2
#Download PDF files. These are then batch processed in bash (text extraction).

urlList <- read.csv("fccData.csv", header=T)

num_recs <- dim(urlList)[1] #num records
urls<- c(as.character(urlList[,3]), as.character(urlList[,4]), as.character(urlList[,5]), as.character(urlList[,6]),as.character(urlList[,7]))
#which urls entries exist (fields not blank)
val_link <- which(urls!=""|urls!=NA)

urls <- urls[val_link] #valid links
row_val_link <- val_link %% num_recs #to keep track of which row (i.e.), date, filer.
col_val_link <- (val_link %/% num_recs)+1

#Deterministic download: save original location reference (row,col) as part of filename.
for (i in 381:420){
	Sys.sleep(3)
	if (length(grep(".pdf",urls[i],ignore.case=T)) == 1){	
		download.file(urls[i], timeout=2, destfile = paste0("./tempdocs/file",as.character(row_val_link[i]),"-",as.character(col_val_link[i]),".pdf"))
	}
	else{cat(paste("Skip download for non PDF file in row ", as.character(i), "and URL", urls[i] )) }	
} 
