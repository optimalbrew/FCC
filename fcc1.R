#exploratory FCC rule making extract file URLs from csv and saving to a new csv

#read in the data set
data <- read.csv(file= "ecfsresults.csv", sep = ",", header=F, skip=1)#, nrows = 10 ) #header is false, num cols data doesn't match num headers

dim(data)
data<-data.frame(data[,c(1,6,9:13)])

colnames(data) <- c("date","filer", "fileURL1", "fileURL2", "fileURL3", "fileURL4", "fileURL5")

#there are some duplicates files, but they are related to different filings, stored in diff folders (could be eliminated later or overwritten by Curl?) 
#data<- data[!duplicated(data),]

#example entry 
#u <- '=HYPERLINK("https://ecfsapi.fcc.gov/file/1041397808761/170413, CTIA Ex Parte.pdf","170413 CTIA Ex Parte.pdf" )'
#out: note "\" added by R before ". Also in this example, url has "," in first part (added by hand, not in actal data set)
#[1] "=HYPERLINK(\"https://ecfsapi.fcc.gov/file/1041397808761/170413, CTIA Ex Parte.pdf\",\"170413 CTIA Ex Parte.pdf\" )"
extractURL <- function(x){
	x<-gsub('\",[[:print:]]*',"",x) #to avoid removing comma inside first URL (if any)
	x <- gsub( '\"',"",x)
	x<- gsub('=HYPERLINK[(]',"",x)
	x<-gsub('[[:space:]]','%20',x)
}

data[,3:6]<-apply(data[,3:6],MARGIN=2,FUN=extractURL)

#If needed to interpret as Dates
#data[,1] <- as.Date(data[,1], format='%m/%d/%Y')

#save for future use.
write.csv(data, file="fccData.csv", row.names=F)

rm(data)
