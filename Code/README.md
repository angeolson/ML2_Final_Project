This folder is organized as follows:

* Benchmark-Result-Images: contains the plots for loss and accuracy by epoch for the benchmark model
* Main-Result-Images: contains the plots for loss and accuracy by epoch for the main model
* Example-Vegetable-Imags: contains the exmaple datapoints used in our final writeup 
* Interpret-Result-Images:contains the resulting images from our main model interpretation 

Beyond the training and testing notebooks as discussed in the main ReadMe, there are additional notebooks here to be run if desired. All notebooks should use the same Linux pathing as described for the training and testing notebooks:
* EDA.py will conduct basic EDA on the data. It will print out results for number of samples by class within the training, validation, and testing data as well as save one image per class for viewing
* Post_Hoc.py will compile the results from the csv prediction files (output of main_testing.py and benchmark_testing.py). It will output a classification report showing the precision, recall, and macro F1 score by class (and for the full model) as well as save confusion matrices that show in a heatmap style for easy interpretability.
