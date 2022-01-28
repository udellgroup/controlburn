
## ControlBurn v0.1.1

This package implements ControlBurn in python. ControlBurn is a feature selection algorithm that uses a weighted LASSO-based feature selection to prune unnecessary features from tree ensembles. The algorithm is efficient and only requires a single training iteration to run.

Tree ensembles distribute feature importance scores evenly amongst groups of correlated features. The average feature ranking of the correlated group is suppressed, which reduces interpretability and complicates feature selection. Like the linear LASSO, ControlBurn assigns all the feature importance of a correlated group of features to a single feature. The algorithm is able to quickly select a subset of important independent features for further analysis.


### Installation

The easiest way to install ControlBurn is through pip.
```sh
!pip install ControlBurn==0.1.1
```


#### Dependencies

ControlBurn works on python 3.7 or above. The following packages are required.

-   numpy (1.20.1)
-   pandas (1.2.4)
-   sklearn (0.24.1)
-   mosek (9.2.47)
-   cvxpy (1.1.13)

### Quick Start
```python
from ControlBurn.ControlBurnModel import ControlBurnClassifier
cb = ControlBurnClassifier(alpha = 0.1)
cb.fit(X,y)
print(cb.features_selected_) #print selected features
print(cb.feature_importances_) #print feature importances

pred = cb.predict(X) #return predictions of polished model using selected features
```
### Reference Paper

ControlBurn: Feature Selection by Sparse Forests B. Liu, M. Xie, and M. Udell  
ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2021
