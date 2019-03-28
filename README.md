# 3DEM-Resolution-Validation

This repository hosts a collection of data and python scripts written to explore the potential of deep learning for validating and estimating the resolution of cryo-electron microscopy (cryo-EM) density maps. Our research findings were published in Molecules 2019, 24(6), 1181; doi: 10.3390/molecules24061181. The article is open access, and available available at https://www.mdpi.com/1420-3049/24/6/1181/htm .

## Getting Started

Our paper explored validation and estimation using two conceptually distinct approaches. The first, not yet avaliable in this repository (as of March 27, 2019), involved the global classification of cryo-EM density maps into high, medium, and low resolution categories. The second, available now, performed local (voxel-wise) resolution classification of cryo-EM density maps into ten resolution categories approximately from 0-1, 1-2, 2-3, ... , 8-9, and 9+ angstroms (1x10^-10 meters). Code for the global classification approaches is available within the global directory and subdirectories, and likewise, local classification code is availble within the local directory and subdirectories.

### Prerequisites

We wrote our code using the python programming language and a host of available libraries including those listed below. We recommend installing these packages as described on the package's accompanying website.

tensorflow <http://tensorflow.org/>
keras <https://keras.io/>
h5py <https://www.h5py.org/>
numpy <http://www.numpy.org/>
mrcfile <https://pypi.org/project/mrcfile/>
sklearn <https://scikit-learn.org/stable/>

### Installing

Install any missing dependencies to your local environment. Then, clone our repository. 

## Deployment

For the local classification approach, the project is run from the command line or python interpreter with
```python
python ./my/local/path/to/resolution_validation.py
```
This script will run the full path of the local classification project. Note, before running the project data in common, monores, and resmap directories must be extracted from the respective split zip archives. First, `resolution_validation.py` will launch data preparation to transform original electron density maps and labels to the appropriate input shape for the network. The processed data are then stored in respective h5py file repositires for test, train and validation sets. Second, the 3D U-Net is constructed. Third, batches from the appropriate set are fed into the network during training, validation, and, finally, evaluation. The third step is repeated for the monores and resmap labels. During model training logs are saved in a timestamped folder. At the end of training the model producing the best weights and biases is saved to disk.


## Authors

* **Todor Kirilov Avramov<sup>1</sup>** - *Initial work* - [GitHub]()
* **Dan Vyenielo<sup>1</sup>** - *Initial work* - [GitHub](https://github.com/dvnlo)

## Acknowledgments

* Josue Gomez-Blanco<sup>2</sup>
* Swathi Adinarayanan<sup>2</sup>
* Javier Vargas<sup>2</sup>
* Dong Si<sup>1</sup>

1 Computing and Software Systems, University of Washington, Bothell, WA 98011, USA

2 Department of Anatomy and Cell Biology, McGill University, Montreal, QC H3A 0C7, Canada