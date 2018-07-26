import re #加载正则表达式库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
#matplotlib inline

#用程序观察前几行的源数据：
#train_data = pd.read_csv('data/train.csv')  #训练数据集
train_data = pd.read_csv('train.csv')  #训练数据集
#test_data = pd.read_csv('data/test.csv') # 验证数据集
test_data = pd.read_csv('test.csv') # 验证数据集
'''
sns.set_style('whitegrid')
train_data.head()
#choose training data to predict age
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
#如果该列是字符串的，就将该列中出现次数最多的字符串赋予空值,mode()函数就是取出现次数最多的元素。
#replace missing value with U0
train_data['Cabin'] = train_data.Cabin.fillna('U0')
# train_data.Cabin[train_data.Cabin.isnull()]='U0'
from sklearn.ensemble import RandomForestRegressor
#choose training data to predict age
age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:,1:]   #取所有行第1个到最后所有数据
Y = age_df_notnull.values[:,0]    #取所有行第0个数据
# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])
train_data.loc[train_data['Age'].isnull(), ['Age']]= predictAges  #loc通过行标签索引行
print (train_data.loc[:,['Age']])
train_data.info()
# 性别与是否生存的关系 Sex
train_data.groupby(['Sex','Survived'])['Survived'].count()
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
# 以下图为不同性别的生存率，可见在泰坦尼克号事故中，还是体现了 Lady First
#plt.show()
# 船舱等级和生存与否的关系 Pclass
train_data.groupby(['Pclass','Survived'])['Pclass'].count()
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
#plt.show()
#不同等级船舱的男女生存率
train_data.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count()
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
#plt.show()
#年龄与存活与否的关系 Age
 #分别分析不同等级船舱和不同性别下的年龄分布和生存的关系：
fig, ax = plt.subplots(1, 2, figsize = (18, 8))
# subplots子图的行数和列数，1行2个图像，figsize参数：指定绘图对象的宽度和高度，单位为英寸.
sns.violinplot("Pclass", "Age", hue="Survived", data=train_data, split=True, ax=ax[0]) #琴形图
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
#plt.show()
#分析总体的年龄分布
plt.figure(figsize=(12,5))
plt.subplot(121)  #1代表行，2代表列，1代表绘制第1个图
train_data['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Num')
plt.subplot(122) #1代表行，2代表列，2代表绘制第2个图
train_data.boxplot(column='Age', showfliers=False) #showefliers是否显示异常值，默认显示,此处不显示
#plt.show()
# 不同年龄下的生存和非生存的分布情况：
facet = sns.FacetGrid(train_data, hue="Survived", aspect=4)
#注：核密度估计，参考：http://www.lifelaf.com/blog/?p=723
#注：hue代表除row，col之外的第三维度，等级，不同的类型不同的颜色,Palette代表调色板
#使用Facet函数创建plot，以“Sex”字段区分等级，aspect=4代表宽度为之前的4倍
facet.map(sns.kdeplot, 'Age', shade=True)      #使用map函数映射kde，以Age作为X轴
facet.set(xlim=(0, train_data['Age'].max()))   #设置x轴的取值范围为0到最大值
facet.add_legend()           #添加图标，印记
#plt.show()
# 不同年龄下的平均生存率
fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
train_data["Age_int"] = train_data["Age"].astype(int)
average_age = train_data[["Age_int", "Survived"]].groupby(['Age_int'], as_index=False).mean()
#groupby 年龄来获取生存率均值，as_index=false代表不用age_int作为新数据表中索引
sns.barplot(x='Age_int', y='Survived', data=average_age)
#plt.show()
# 输出年龄特征的统计信息
print(train_data['Age'].describe())
bins = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'], bins)
by_age = train_data.groupby('Age_group')['Survived'].mean()
by_age.plot(kind='bar')
#plt.show()
# 乘客的姓名称呼与存活与否的关系 Name
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#正则表达式，匹配任何字母，包括大写和小写的， +代表匹配一个字母后面接着0或者多个字母，\.代表匹配0或1个连续的点
##参数expand=False在一组返回值的情况下,返回序列(Series)
print(pd.crosstab(train_data['Title'], train_data['Sex']))
#交叉表是用于统计分组频率的特殊透视表
#观察不同称呼与生存率的关系
train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()
#plt.show()
#同时，对于名字，我们还可以观察名字长度和生存率之间存在关系的可能
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Name_length'] = train_data['Name'].apply(len)
name_length =train_data[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()
sns.barplot(x='Name_length', y='Survived', data=name_length)
#plt.show()
# 有无兄弟姐妹和存活与否的关系 SibSp
# 将数据分为有兄弟姐妹的和没有兄弟姐妹的两组
sibsp_df = train_data[train_data['SibSp'] != 0]
no_sibsp_df = train_data[train_data['SibSp'] == 0]
plt.figure(figsize=(10,5))
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('sibsp')
plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_sibsp')
#plt.show()
#有无父母子女和存活与否的关系 Parch，和有无兄弟姐妹一样，同样分析可以得到：
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]
plt.figure(figsize=(10,5))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived','Survived'],autopct='%1.1f%%')
plt.xlabel('parch')
plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No survived','Survived'],autopct='%1.1f%%')
plt.xlabel('no_parch')
#plt.show()
#亲友的人数和存活与否的关系 SibSp & Parch
fig,ax=plt.subplots(1,2,figsize=(18,8))
train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')
train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
train_data[['Family_Size','Survived']].groupby(['Family_Size']).mean().plot.bar()
#plt.show()
# 票价分布和存活与否的关系 Fare, 首先绘制票价的分布情况：
plt.figure(figsize=(10,5))
train_data['Fare'].hist(bins = 70)
train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
#plt.show()
train_data['Fare'].describe()  #查看 Fare 的数据分布
#绘制生存与否与票价均值和方差的关系：
fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]
average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
average_fare.plot(yerr=std_fare, kind='bar', legend=False)
# 把 Cabin 缺失值替换为 "U0"
train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U0'
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()
#plt.show()
#对不同类型的船舱进行分析：
# create feature for the alphabetical part of the cabin number
train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
# convert the distinct cabin letters with incremental integer values
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()
# 港口和存活与否的关系 Embarked
# 泰坦尼克号从英国的南安普顿港出发，途径法国瑟堡和爱尔兰昆士敦，那么在昆士敦之前上船的人，有可能在瑟堡或昆士敦下船，这些人将不会遇到海难
sns.countplot('Embarked', hue='Survived', data=train_data)
plt.title('Embarked and Survived')
#分析得出在不同的港口上船，生还率不同，C 最高，Q 次之，S 最低
sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)
plt.title('Embarked and Survived rate')
#plt.show()
'''

#变量转换,虚拟变量
embark_dummies  = pd.get_dummies(train_data['Embarked'])
train_data = train_data.join(embark_dummies)        #生成新的一列
train_data.drop(['Embarked'], axis=1,inplace=True)
embark_dummies = train_data[['S', 'C', 'Q']]
embark_dummies.head()
# Replace missing values with "U0"
train_data['Cabin'][train_data.Cabin.isnull()] = 'U0'
# create feature for the alphabetical part of the cabin number
train_data['CabinLetter'] = train_data['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())
# convert the distinct cabin letters with incremental integer values
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
train_data['CabinLetter'].head()

#下面对 Age 进行 Scaling
from sklearn import preprocessing
assert np.size(train_data['Age']) == 891
# StandardScaler will subtract the mean from each value then scale to the unit variance
scaler = preprocessing.StandardScaler()
train_data = train_data.dropna()
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
#reshape重新调整矩阵的行数，列数，维度。
train_data['Age_scaled'].head()
#  定量(Quantitative)转换：Binning 分箱
# Divide all fares into quartiles
train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 5)
train_data['Fare_bin'].head()
fare_bin_dummies_df = pd.get_dummies(train_data['Fare_bin']).rename(columns=lambda x: 'Fare_'+ str(x))
train_data = pd.concat([train_data, fare_bin_dummies_df], axis=1)
#axis 1表示横轴，方向从左到右；0表示纵轴，方向从上到下。当axis=1时，数组的变化是横向的，而体现出来的是列的增加或者减少。
#train_df_org = pd.read_csv('data/train.csv')
train_df_org = pd.read_csv('train.csv')
#test_df_org = pd.read_csv('data/test.csv')
test_df_org = pd.read_csv('test.csv')
test_df_org['Survived'] = 0
combined_train_test = train_df_org.append(test_df_org)
PassengerId = test_df_org['PassengerId']
#因为“Embarked”字段的缺失值不多，所以这里我们以众数来填充：
combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
#inplace=true不创建新的对象，在原对象上进行修改,iloc[0]根据行号索引
combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
# 使用 pd.get_dummies 获取 one-hot 编码
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)
#对 Sex 也进行 one-hot 编码，也就是 dummy 处理：
# 为了后面的特征分析，这里我们也将 Sex 特征进行 facrorizing
combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)
# Name 字段，首先先从名字中提取各种称呼：
combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
#将各式称呼进行统一化处理：
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
#使用 dummy 对不同的称呼进行分列
# 为了后面的特征分析，这里我们也将 Title 特征进行 facrorizing
combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)
#增加名字长度的特征：
combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)
# Fare 字段，由前面分析可以知道，Fare 项在测试数据中缺少一个值，所以需要对该值进行填充。
#我们按照一二三等舱各自的均价来填充：
#下面 transform 将函数 np.mean 应用到各个 group 中。
combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
#通过对 Ticket 数据的分析，我们可以看到部分票号数据有重复，同时结合亲属人数及名字的数据，和票价船舱等级对比，我们可以知道购买的票中有家庭票和团体票，所以我们需要将团体票的票价分配到每个人的头上。
combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)
#使用 binning 给票价分等级
combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
#对于 5 个等级的票价我们也可以继续使用 dummy 为票价等级分列：
combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))
combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_df], axis=1)
combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)


#对原数组直接修改为【Fare_bin】
from sklearn.preprocessing import LabelEncoder

# 建立 PClass Fare Category
def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'
Pclass1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]
Pclass2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]
Pclass3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]

# 建立 Pclass_Fare Category
combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)

pclass_level = LabelEncoder()
#多个实参，放到一个元组里面,以*开头，可以传多个参数；**是形参中按照关键字传值把多余的传值以字典的方式呈现.*args：（表示的就是将实参中按照位置传值，多出来的值都给args，且以元祖的方式呈现）
# 给每一项添加标签
pclass_level.fit(np.array( ['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))
# 转换成数值
combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])
# dummy 转换
pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(columns=lambda x: 'Pclass_' + str(x))
combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)
#同时，我们将 Pclass 特征 factorize 化：
combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]
#Parch and SibSp 字段
#由前面的分析知道，亲友的数量没有或者太多会影响到 Survived。所以将二者合并为 FamliySize 这一组合项，同时也保留这两项。
def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'
combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)
le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])
family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],prefix=combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)


# 这里我们使用后者来处理。以 Age 为目标值，将 Age 完整的项作为训练集，将 Age 缺失的项作为测试集。
missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size','Family_Size_Category', 'Fare', 'Fare_bin_id', 'Pclass']])
missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
missing_age_test.head()
#建立 Age 的预测模型，我们可以多模型预测，然后再做模型的融合，提高预测的精度。
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
    # model 1  gbm
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    # model 2 rf
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])
    # two models merge
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
    # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)
    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])
    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)
    return missing_age_test
#利用融合模型预测的结果填充 Age 的缺失值：
if __name__=='__main__':
    combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)


#Ticket 字段
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)
# 如果要提取数字信息，则也可以这样做，现在我们对数字票单纯地分为一类。
# combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x:pd.to_numeric(x, errors='coerce'))
# combined_train_test['Ticket_Number'].fillna(0, inplace=True)
# 将 Ticket_Letter factorize
combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]
# Cabin 字段m暂时保留该特征，并将其分为有和无两类。
combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
#我们挑选一些主要的特征，生成特征之间的关联图，查看特征与特征之间的相关性：
Correlation = pd.DataFrame(combined_train_test[['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass','Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
combined_train_test.head()
#特征之间的数据分布图


#输入模型前的一些处理
#一些数据的正则化：这里我们将 Age 和 fare 进行正则化
from sklearn import preprocessing
#把空值删掉
combined_train_test = combined_train_test.dropna()
scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age','Fare','Name_length']])

combined_train_test[['Age','Fare','Name_length']] = scale_age_fare.transform(combined_train_test[['Age','Fare','Name_length']])

combined_data_backup = combined_train_test
combined_train_test.drop(['PassengerId','Embarked','Sex','Name','Fare_bin_id','Pclass_Fare_Category',
                          'Parch','SibSp','Family_Size_Category','Ticket'],axis=1,inplace=True)

train_data = combined_train_test[:891]
test_data = combined_train_test[891:]

titanic_train_data_X = train_data.drop(['Survived'], axis=1)
titanic_train_data_Y = train_data['Survived']
titanic_test_data_X = test_data.drop(['Survived'], axis=1)


'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    # randomforest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Feeatures from RF Classifier')
    print(str(features_top_n_rf[:10]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best DT Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Bset DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # merge the three models
    features_top_n = pd.concat(
        [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
        ignore_index=True).drop_duplicates()
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                     feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)

    return features_top_n, features_importance
'''

