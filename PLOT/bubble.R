library(ggplot2)
library(RColorBrewer)
library(reshape2)
args=commandArgs(T)
parameter1 = args[1]
parameter2 = args[2]
data.final<-read.csv(parameter1,header=T,row.names = 1,check.names=F)
data.final<-as.matrix(data.final)
mydata <- melt(data.final)
colnames(mydata)<-c("Cell_type1","Cell_type2","value")
pdf(file = parameter2,width = 9,height = 8)
ggplot(mydata, aes(x= Cell_type1 , y=Cell_type2)) +theme_bw()+
    geom_point(aes(size=value,fill = value), shape=21, colour="black") +
    scale_fill_gradientn(colours=c(brewer.pal(7,"Blues")[7],brewer.pal(7,"Reds")[6]),na.value=NA)+
    scale_size_area(max_size=14, guide = "none") +
    #geom_text(aes(label=value),color="white",size=5) +
    theme(panel.grid = element_blank(),
          text=element_text(size=13,face="plain",color="black"),axis.text.x=element_text(angle=45,hjust =1,vjust=1),
          axis.title=element_text(size=0,face="plain",color="white"),
          axis.text = element_text(size=23,face="plain",color="black"),
          legend.position="right"
    )
while (!is.null(dev.list()))  dev.off()




