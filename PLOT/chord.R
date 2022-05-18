suppressMessages(library(RColorBrewer))
suppressMessages(library(ggalluvial))
suppressMessages(library(ggplot2))
suppressMessages(library(circlize))
options(warn = -1)

args=commandArgs(T)
parameter1 = args[1]
parameter2 = args[2]
#parameter3 = args[3]
#grid_col = c("red", "green", "cornflowerblue","blueviolet", brewer.pal(8, 'Accent')[6], "darkseagreen1", "hotpink1", "gold", "slateblue3", brewer.pal(8, 'Set3')[5],brewer.pal(8, 'Set3')[4])
#grid_col = parameter3
#print(grid_col)
pdf(file = parameter1,width = 10,height = 10)

mat <- read.csv(file = parameter2, row.names = 1,head=T)
colnames(mat) <- rownames(mat)
y = data.matrix(mat)
#circos.par( clock.wise = FALSE,cex = 2)
#par(cex = 1.5)
chordDiagram(y,directional = 1, 
             direction.type = c("arrows"),
             link.arr.type = "triangle",big.gap = 40, small.gap = 10,annotationTrack = c("name", "grid"),annotationTrackHeight = c(0.03, 0.06))
while (!is.null(dev.list()))  dev.off()

