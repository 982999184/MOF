### Configuration
Code is tested under the environment of Pytorch 1.9.0, Python 3.8 and CUDA 11.1 on Windows. 

Data: The data for this study are presented by Paper [High-Performing Deep Learning Regression Models for Predicting
Low-Pressure CO2Adsorption Properties of Metal−Organic
Frameworks](https://doi.org/10.1021/acs.jpcc.0c06334) and can be downloaded [here](https://1drv.ms/u/s!AtuVqcWZi8aAy11S2wxataTe8IMH).

### Usage
#### Direct use of pre training model
Modify 'path' in line 11 of ```predict_list.py``` to be predicted CIF path, click run to predict, The predictions will be generated in the root directory in the form of a table. To modify the prediction target, change line 18 in ```predict_list.py``` to 'model_sort_cap.pht'
#### Training tasks from scratch
Modify 'path' in line 8 of ```CIF_to_npy.py``` to be predicted CIF path, click run to project CIF files. Modify 'train_path' in line 15 of ```train.py``` to be npy file path, click Run to train. The code will automatically save the best model in a directory.
