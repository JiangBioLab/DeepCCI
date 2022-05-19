#Rscript pbmc.R &>nohup.out&
suppressMessages(library(CellChat))
suppressMessages(library(patchwork))
options(warn = -1)
options(stringsAsFactors = FALSE)
future::plan("multiprocess", workers = 6)
options(future.globals.maxSize = 2000 * 1024^2)
suppressMessages(require(tibble))
suppressMessages(require(magrittr))
suppressMessages(require(purrr))

suppressMessages(library(Seurat))
suppressMessages(library(dplyr))
warnings('off')
args=commandArgs(T)
parameter1 = args[1]
parameter2 = args[2]
parameter3 = args[3]
testdata <- readRDS(parameter1)

data.input <- GetAssayData(testdata, assay = "RNA", slot = "data")
label <- read.csv(parameter2)
labels <- as.factor(label$labels)
meta <- data.frame(labels = labels, row.names = names(labels))

LRDB<-load(parameter3)

load('./LRDB/myCompute.RData')
#load('./LRDB/myCompute1.RData')
test <- suppressMessages(createCellChat(object = data.input, meta = meta, group.by = "labels"))

test <- suppressMessages(addMeta(test, meta = meta))
test <- suppressMessages(setIdent(test, ident.use = "labels"))
#levels(cellchat@idents) 
groupSize <- as.numeric(table(test@idents))

HumanDB <- LRDB.human

test@DB <- HumanDB

test <- suppressMessages(subsetData(test))

test <- suppressMessages(identifyOverExpressedGenes(test))
test <- suppressMessages(identifyOverExpressedInteractions(test))

test <- suppressMessages(projectData(test, PPI.human))

test <- suppressMessages(mycomputeCommunProb(test))
df.net <- suppressMessages(subsetCommunication(test,thresh = 1))

write.csv(test@LR$LRsig,"./output/pairLR_use.csv")
write.csv(test@meta,"./output/meta.csv")
write.csv(test@data.project,"./output/data_project.csv")
write.csv(test@DB$complex,"./output/complex_input.csv")
write.csv(test@DB$cofactor,"./output/cofactor.csv")
write.csv(df.net,"./output/df_net.csv")
write.csv(test@DB$interaction,"./output/pairLR.csv")

