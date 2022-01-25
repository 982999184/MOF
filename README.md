### Configuration
Code is tested under the environment of Pytorch 1.9.0, Python 3.8 and CUDA 11.1 on Windows. 

Data: The data for this study are presented by Paper [High-Performing Deep Learning Regression Models for Predicting
Low-Pressure CO2Adsorption Properties of Metal−Organic
Frameworks](https://doi.org/10.1021/acs.jpcc.0c06334) and can be downloaded [here](https://1drv.ms/u/s!AtuVqcWZi8aAy11S2wxataTe8IMH).

### Usage
+ Download the data and use the ```CIF_to_npy.py``` file to project the data from CIF format to npy format. where path is the storage path of the CIF file and save_path is the storage path of the npy file.
    
+ Use ```train.py``` to train the model, the trained model parameter files will be stored in the root directory.

+ The model can be predicted using ```predict_list.py``` after training. ```model_sort_cap.pht``` given in this file is a pre-trained model for predicting the performance of CO2 working capacity using sort projection method.

### Modification of the model
+ When the prediction target needs to be modified, modify line 27 in ```train.py``` and line 120 in ```predict_list.py``` .
+ When projection weights need to be modified, modify line 42 in ```CIF_to_npy.py``` and line 52 in ```predict_list.py```. It is worth noting that the weights in CIF_to_npy should be the same as in predict_list.