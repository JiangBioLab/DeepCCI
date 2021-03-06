suppressMessages(library(Seurat))
suppressMessages(library(scCATCH))

args=commandArgs(T)
parameter1 = args[1]
options(warn = -1)
testdata <- readRDS(parameter1)
testdata <-Seurat::NormalizeData(testdata)
testdata <-Seurat::FindVariableFeatures(testdata,selection.method = "vst", nfeatures = 2000)
testdatasc <- rev_gene(data = testdata@assays$RNA@data, data_type = "data", species = "Human", geneinfo = geneinfo)
label<-read.csv('./output/pre_label.csv')
labels<-as.character(label$labels)
obj <- createscCATCH(data = testdatasc, cluster = labels)
obj <- findmarkergene(object = obj, species = "Human", marker = cellmatch, tissue = c('Blood','Peripheral blood','Bone marrow'))
obj <- findcelltype(object = obj)
print(obj@celltype$cell_type)
write.csv(obj@celltype$cell_type,"./output/cell_type.csv")
write.csv(obj@celltype$cluster,"./output/cell_cluster.csv")