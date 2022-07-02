# DeepCCI (Deep learning framework for Cell-Cell Interactions inference from scRNA-seq data)

DeepCCI is a graph convolutional network (GCN)-based deep learning framework for Cell-Cell Interactions inference from scRNA-seq data.
![workflow](https://user-images.githubusercontent.com/72069543/169433397-ff34dce1-717f-446e-8b0a-0e1b5ccf6da6.png)


## Installation:

### From Source:

Start by grabbing this source codes:

```
git clone https://github.com/JiangBioLab/DeepCCI.git
cd DeepCCI
```

###  (Recommended) Use python virutal environment with conda（https://anaconda.org/）

```
conda create -n deepcciEnv python=3.7.4 pip
conda activate deepcciEnv
pip install -r requirements.txt
```
and then also install the following in R:

```
conda install r-base
R
data = load('Rpack.Rdata')
install.packages(c(Rpack))
```

### 
### Quick Start

### 1. Cell Cluster Model

#### **(1) Preprocess input files**

The cluster model of DeepCCI accepts scRNA-seq data format: CSV and h5. The processed feature file of scRNA-seq data will be provided. Depending on the size of the scRNA-seq file，the process will take 5-10 minutes.

##### CSV format

Take an example of Yan 's  (GSE36552). 

```
cd Cluster_model
python preprocess.py --name Yan --file_format csv
```

##### h5 format

Take an example Qx Limb Muscle (GSE109774 ).

```
cd Cluster_model
python preprocess.py --name  Quake_10x_Limb_Muscle --file_format h5
```

#### (2) Cell Clustering
The clustering results of scRNA-seq data will be output. 
##### With pre-train:
It will take 25 minutes.
```
python Cluster.py --name Yan --pretain True --pretrain_epoch 50 --device cuda
```

Without pre-train:
The pretrained model files are in the pretain_model folder.
It will take 5 minutes.
```
python Cluster.py --name Yan --pretain False --device cuda
python Cluster.py --name Quake_10x_Limb_Muscle --pretain False --device cuda
```

### 2. Cell Interaction Model

#### **(1) Preprocess input files**

The example test file can be download from  http://jianglab.org.cn/deepcci_download/.
The processed feature file will be provided. Depending on the size of the scRNA-seq file，the process will take 10-20 minutes.

##### With cell-label:

```
cd Interaction_model
python Feature.py --label_mode True --species Human
```

##### Without cell-label

```
python Feature.py --label_mode False --species Human
```

#### (2) Interaction Inference
The predicted interaction outfile will be provided. The predicted process will take 1-2 minutes.
```
python Interaction_inference.py --device cuda 
```

### 3. Visualization

#####
To show the CCI output intuitively, several visualization methods are provided. The process will take 1 minutes.
```
cd Plot
python Plot.py
```

## Contact

Feel free to submit an issue or contact us at wenyiyang22@163.com for problems about the package.
