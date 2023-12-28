# CHD
A machine learning-based model for risk stratification of coronary heart disease has been created by our team and is now available for download. You can, of course, download the [CCRS](https://pypi.org/project/CCRS/) package as well. In the future, we might carry on with iterations.

![CHD Figure](https://www.cdc.gov/heartdisease/images/coronary-artery-disease-medium.jpg)

A simple example

```python
import pickle
import scipy.stats as st
import pandas as pd
import numpy as np
import imblearn
import CCRS
loaded_model = pickle.load('./data/CHD_RF_lessFea.pkl')
Traindata = pd.read_csv('./data/CHD_age.csv')
# Naturally, the test data in this case can be replaced with the user's own data.
X_test = Traindata.iloc[:5]
scores = loaded_model.predict_proba(X_test)
value = scores[:,1]
CHD_risk_group = risk_label(value)
df = pd.DataFrame({'y_pred': scores[:,1],'Age':X_test.Age})
CHD_RRS = RRScore(df,Traindata,mean=52.8,sd=9.8)
result = pd.DataFrame({'CHD risk group': CHD_risk_group,'CHD relative risk rank':CHD_RRS})
```

Output:

| CHD risk group | CHD relative risk rank |
| -----------    | -----------            |
| mild           | 83.5%                  |
| moderate       | 56.1%                  |
| mild           | 79.1%                  |
| mild           | 86.8%                  |
| severe         | 95.4%                  |


How to cite:

Chen, B., Ruan, L., Yang, L., Zhang, Y., Lu, Y., Sang, Y., Jin, X., Bai, Y., Zhang, C., and Li, T. (2022). Machine learning improves risk stratification of coronary heart disease and stroke. Ann Transl Med 10, 1156. 10.21037/atm-22-1916.
  
