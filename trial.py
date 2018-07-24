'''
Age字段，因为Age项的缺失值较多，所以不能直接填充age的众数或者平均数。
常见的有两种对年龄的填充方式：一种是根据Title中的称呼，如Mr，Master、Miss等称呼不同类
别的人的平均年龄来填充；一种是综合几项如Sex、Title、Pclass等其他没有缺失值的项，使用机器
学习算法来预测Age。

这里我们使用后者来处理。以Age为目标值，将Age完整的项作为训练集，将Age缺失的项作为
测试集。
'''
# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing


combined_train_test=pd.concat([combined_train_test,family_size_dummies_df],axis=1)
missing_age_df=pd.DataFrame(combined_train_test[['Age','Embarked','Sex','Title','Name_length','Family_Size',\
                                                 'Family_Size_Category','Fare','Fare_bin_id','Pclass']])
print(missing_age_df)

'''
missing_age_train=missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test=missing_age_df[missing_age_df['Age'].isnull()]
missing_age_test.head()
#输入结果：
AgeEmbarked
SexTitleName_length
Family_SizeFamily_Size_CategoryFareFare_bin_idPclass
5
NaN
2
0
0
161
1
8.45832
0
17NaN
0
0
0
281
1
13.00003
2
19NaN
1
1
1
231
1
7.22504
0
26NaN
1
0
0
231
1
7.22504
0
'''