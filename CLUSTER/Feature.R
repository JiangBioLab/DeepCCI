suppressMessages(library(Seurat))
args=commandArgs(T)
parameter1 = args[1]
parameter2 = args[2]
options(warn = -1)

testdata <- readRDS(parameter1)
testdata <-Seurat::NormalizeData(testdata)
testdata <-Seurat::FindVariableFeatures(testdata,selection.method = "vst", nfeatures = 2000)
data.input<-testdata@assays$RNA@data[testdata@assays$RNA@var.features,]
write.csv(data.input,"./output/Top2000.csv")



#grid_col <-  c("red", "green", "cornflowerblue","blueviolet", brewer.pal(8, 'Accent')[6], "darkseagreen1", "hotpink1", "hotpink4", "gold", "slateblue3","tomato", brewer.pal(8, 'Set3')[5],brewer.pal(8, 'Set3')[4])

#brewer.pal(8, 'Reds')[6],brewer.pal(8, 'YlGn')[5],brewer.pal(8, 'YlGnBu')[6],brewer.pal(8, 'RdPu')[7],brewer.pal(8, 'Purples')[6],
#brewer.pal(8, 'Reds')[5],brewer.pal(8, 'PuRd')[7],brewer.pal(8, 'PuBuGn')[7],brewer.pal(8, 'PuBu')[8],brewer.pal(8, 'Oranges')[5],
#brewer.pal(8, 'BuPu')[8],brewer.pal(9, 'Oranges')[9],brewer.pal(8, 'Blues')[8]

