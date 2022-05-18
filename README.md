# DeepCCI (Deep learning framework for Cell- Cell Interactions inference from scRNA-seq data)

DeepCCI is a graph convolutional network (GCN)-based deep learning framework for Cell- Cell Interactions inference from scRNA-seq data.
![workflow](https://user-images.githubusercontent.com/72069543/169000112-e834cf30-082b-4c7f-b722-377c4daffed9.png)

## Installing the Python package

Installation within virtual environments are recommended, see [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda](https://conda.io/docs/user-guide/tasks/manage-environments.html).

For conda, here's a one-liner to set up an empty environment for installing DeepCCI:

```
conda create -n cb python=3.6 && conda activate cb
```

## Upcoming changes

The Python package will migrate to using [anndata](https://anndata.readthedocs.io/en/latest/index.html) as the data class.

## Reproduce results in the paper

To reproduce results, please check out the `rep` branch.

## Contact

Feel free to submit an issue or contact us at wenyiyang22@163.com for problems about the Python package, website or database.
