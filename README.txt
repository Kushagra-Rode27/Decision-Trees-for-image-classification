Name - Kushagra Rode , Entry Number - 2020CS10354

The code is modularised into multiple files
1. feat.py - This contains 3 functions load_images, load_test and load_multi. These are used to extract the features 
            and their labels from the training, validation and test set

2. dt.py - This contains my own implemetation of decision tree from scratch. It supports two criterion - gini index
           and information gain

3. sklearn_dt.py - This file contains all the sections which are created and trained using the sklearn library. Used for 
                   binary classification

4. sklearn_multi.py - Similar to sklearn_dt.py, only difference is that this file is used for multi-class classification

5. main_binary.py - This file generates the predicted outputs for models of each of the sections for binary classification. 
                    The outputs are generated in their respective .csv files.

6. main_multi.py - Similar to the main_binary.py, but this generates predicted outputs for models of each of the sections 
                   for multi-class classification.

To run the main_binary.py, run the following code :

-> python main_binary.py --train_path="./train" --test_path="./test_sample" --out_path="./output"

Here train_path, test_path are folders to the training and test data respectively.Make sure the out_path is the location of the 
folder where you want all the .csv files to be stored.

For running main_multi.py, run the following code :

-> python main_multi.py --train_path="./train" --test_path="./test_sample" --out_path="./output"
