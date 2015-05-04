# manipulation images experiment

### occludeness classification

data set: segmented, binary. (48 by 64 grayscale).

splitting: random 3,1,1

network architecture: one layer hidden neural nets.

#### 3 class (wo object/ occluded/ with object)

| classifiers | train | valid | test|
| ------------ | ------------- | ------------ |------------ |
| logistic regression | -   | **56.7** |63.3|
| hidden = 100 | 53.6  | 61.3 | 64.7|
| hidden = 500 | 50.2  | **58.0** | 60.7|
| hidden = 1000 | 54.5  | 62.0 |62.7|
| hidden = 2000 | 48.2 | 59.3 |63.3|

#### 2 class (with object vs. the rest)

| classifiers | train | valid | test|
| ------------ | ------------- | ------------ |------------ |
| logistic regression | -   | **41.33** |40.0|
| hidden = 100 | 39.09  | 41.33 | 40.00|
| hidden = 500 | 39.09  | 41.33 | 40.00|
| hidden = 1000 | 39.09  | 41.33 |40.00|
| hidden = 2000 | 37.45 | **40.67** |40.00|


### reconstruction 

on **validation** set, the orginal images and reconstructed images are as follows,
(with hidden layer size 100)

![Alt text](rec/orig_va1.png =300x)
![Alt text](rec/rec_va1.png =300x)

![Alt text](rec/orig_va2.png =300x)
![Alt text](rec/rec_va2.png =300x)


