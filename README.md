### MachineLearning
Sima Esfandiarpour Borujeni


**Project Title**: 
Modeling Fluid Dynamics Using Machine Learning Techniques

**Project Idea:**
In this project, we will perform multiple classification & regression techniques and also will be operating on 2 different data sets.

__*The first data set*__ contains 13910 measurements from 16 chemical sensors utilized in simulations for drift compensation in a discrimination task of 6 gases at various levels of concentrations from 5 to 1000 ppmv. The output from the 16 sensors must be used to determine the concentration levels of the 6 gases.
The resulting dataset comprises recordings from six distinct pure gaseous substances, namely Ammonia, Acetaldehyde, Acetone, Ethylene, Ethanol, and Toluene, each dosed at a wide variety of concentration values ranging from 5 to 1000 ppmv.

We use the following cross validation parameters:

Batch |	C	|Gamma |(É¤)	Rate
--------|----|--------|--------
1|	256.0	|0.03125	|98.8764
2	|64.0|	0.00390625|	99.7588
3	|128.0|	0.03125|	100.0
4	|1.0	|0.2 | 100.0
5|	2.0	|0.015625|	99.4924
6	|256.0|	0.0009765625	|99.5217
7	|64.0	|0.0625	 | 99.9723
8|	1024.0	|0.0078125	|99.6599
9|	2.0	|0.00390625	|100.0

__*The second data set *__ contains 14 measurements on various data concerning the state of a gas turbine used in a ship (ship speed, propeller torque, compressor pressure, etc). These data points correspond with two coefficients of the system’s level of degradation; one for the compressor, and one for the turbine. Our objective is to find the best classifier for each data set.

**Datasets:**

The first data set is: Gas Sensor Array Drift Data Set 
**(a) Classification: Comparing the performance of the following classification models**

1. Support Vector Machines (also choose different kernels - linear, RBF)
2. Logistic Regression
3. Decision Trees
4. Random Forests
5. k - Nearest neighbor
6. Naive Bayes

Data link- https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset



The second data set is : Condition Based Maintenance of Naval Propulsion Plants Data Set 
**(b) Regression: Comparing the performance of the following regression models**
1. Support vector regression (choose different kernels)
2. Nearest Neighbor
3. Decision Tree
4. Gaussian Process
5. Generalized linear regression models (Ridge Regression, Lasso Ridge Regression, Linear Regression, Bayesian Regression)

Data link - https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants




