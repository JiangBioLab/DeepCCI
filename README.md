# DeepCCI (Deep learning framework for Cell-Cell Interactions inference from scRNA-seq data)

DeepCCI is a graph convolutional network (GCN)-based deep learning framework for Cell- Cell Interactions inference from scRNA-seq data.
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
for (i in Rpack) install.packages()
```

### 
### Quick Start

### 1. Cell Cluster Model

#### **(1) Preprocess input files**

The cluster model of DeepCCI accepts scRNA-seq data format: CSV and h5

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

##### With pre-train:

```
python Cluster.py --name Yan --pretain True --pretrain_epoch 50 --device cuda
```

Without pre-train:

```
python Cluster.py --name Yan --pretain False --device cuda
```

### 2. Cell Interaction Model

#### **(1) Preprocess input files**

The example test file can be download from  http://jianglab.org.cn/deepcci_download/

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

```
python Interaction_inference.py --device cuda 
```

### 3. Visualization

##### With cell-label:

```
cd Plot
python Plot.py
```

## Contact

Feel free to submit an issue or contact us at wenyiyang22@163.com for problems about the Python package, website or database.
