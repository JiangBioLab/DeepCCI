suppressMessages(library(psych))
suppressMessages(library(qgraph))
suppressMessages(library(igraph))
suppressMessages(library(purrr))
suppressMessages(library(RColorBrewer))
args=commandArgs(T)
parameter1 = args[1]
parameter2 = args[2]
options(warn = -1)
netf<- read.csv(parameter1,header=T,row.names = 1,check.names=F)
mynet <- netf
net<- graph_from_data_frame(mynet)

allcolour = c("red", "green", "cornflowerblue","blueviolet", brewer.pal(8, 'Accent')[6], "darkseagreen1", "hotpink1", "gold", "slateblue3", brewer.pal(8, 'Set3')[5],brewer.pal(8, 'Set3')[4])
pdf(file = parameter2,width = 12,height = 12)

karate_groups <- cluster_optimal(net)
coords <- layout_in_circle(net, order =
                             order(membership(karate_groups)))

E(net)$width  <- E(net)$count*1
V(net)$color <- allcolour
E(net)$color <- tail_of(net, E(net))$color
plot(net, edge.arrow.size=.5, 
     edge.curved=0,
     #edge.color=allcolour,
     #vertex.color=allcolour,
     vertex.frame.color="#555555",
     vertex.label.color="black",
     layout = coords,
     vertex.size = 30,
     vertex.label.cex=2)
while (!is.null(dev.list()))  dev.off()

