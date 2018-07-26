# -*- coding: utf-8 -*-
import TitanicData as Td
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from sklearn.learning_curve import learning_curve

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

train_data = pd.read_csv('train.csv')  # 训练数据集
test_data = pd.read_csv('test.csv')  # 验证数据集

Td.fare_binning(train_data)