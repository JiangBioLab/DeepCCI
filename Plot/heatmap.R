library(ggplot2)  
library(RColorBrewer)  
library(reshape2)
library(pheatmap)

args=commandArgs(T)
parameter1 = args[1]
parameter2 = args[2]
pdf(file = parameter1,width = 12,height = 5)

data.final<-read.csv(parameter2,header=T,row.names = 1,check.names=F)
pheatmap((data.final),cluster_rows = FALSE, cluster_cols  = FALSE,cellheight = 8, cellwidth = 8,
         cexCol = 1,angle_col  = "45",fontsize=6)
while (!is.null(dev.list()))  dev.off()