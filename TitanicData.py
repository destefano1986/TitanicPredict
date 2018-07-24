# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

warnings.filterwarnings('ignore')

def titanic_plot_pie(df):
    df['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
    plt.show()

def missing_value_process(df):
    df.Embarked[train_data.Embarked.isnull()] = df.Embarked.dropna().mode().values
    df['Cabin'] = df.Cabin.fillna('U0')  # train_data.Cabin[train_data.Cabin.isnull()]='U0'
    return df

def missing_value_rf(df):
    # choosetrainingdatatopredictage
    age_df = df[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[(df['Age'].notnull())]
    age_df_isnull = age_df.loc[(df['Age'].isnull())]
    X = age_df_notnull.values[:, 1:]
    Y = age_df_notnull.values[:, 0]
    # useRandomForestRegressiontotraindata
    RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    RFR.fit(X, Y)
    predictAges = RFR.predict(age_df_isnull.values[:, 1:])
    df.loc[df['Age'].isnull(), ['Age']] = predictAges
    return df

def sex_survival_relation(df):
    df.groupby(['Sex', 'Survived'])['Survived'].count()
    df[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
    plt.show()

def pclass_survival_relation(df):
    df.groupby(['Pclass', 'Survived'])['Pclass'].count()
    df[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar()
    plt.show()

def age_distribution(df):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    df['Age'].hist(bins=70)
    plt.xlabel('Age')
    plt.ylabel('Num')
    plt.subplot(122)
    df.boxplot(column='Age', showfliers=False)
    plt.show()

def pclass_sex_survival_relation(df):
    df.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count()
    df[['Sex', 'Pclass', 'Survived']].groupby(['Pclass', 'Sex']).mean().plot.bar()
    plt.show()

def age_survival_relation(df):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.violinplot("Pclass", "Age", hue="Survived", data=df, split=True, ax=ax[0])
    ax[0].set_title('PclassandAgevsSurvived')
    ax[0].set_yticks(range(0, 110, 10))
    sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])
    ax[1].set_title('SexandAgevsSurvived')
    ax[1].set_yticks(range(0, 110, 10))
    plt.show()

def age_survival_facet(df):
    facet = sns.FacetGrid(df, hue="Survived", aspect=4)
    facet.map(sns.kdeplot, 'Age', shade=True)
    facet.set(xlim=(0, df['Age'].max()))
    facet.add_legend()
    plt.show()

def age_survival_prob(df):
    fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
    df.dropna(inplace=True)
    df["Age_int"] = df["Age"].astype(int)
    average_age = df[["Age_int", "Survived"]].groupby(['Age_int'], as_index=False).mean()
    sns.barplot(x='Age_int', y='Survived', data=average_age)
    plt.show()

def title_survival(df):
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    pd.crosstab(df['Title'], df['Sex'])
    # 观察不同称呼与生存率的关系
    df[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()
    plt.show()

def name_length_survival(df):
    fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
    df['Name_length'] = df['Name'].apply(len)
    name_length = df[['Name_length', 'Survived']].groupby(['Name_length'], as_index=False).mean()
    sns.barplot(x='Name_length', y='Survived', data=name_length)
    plt.show()

def siblings_survival(df):
    sibsp_df = df[df['SibSp'] != 0]
    no_sibsp_df = df[df['SibSp'] == 0]
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    sibsp_df['Survived'].value_counts().plot.pie(labels=['NoSurvived', 'Survived'], autopct='%1.1f%%')
    plt.xlabel('sibsp')
    plt.subplot(122)
    no_sibsp_df['Survived'].value_counts().plot.pie(labels=['NoSurvived', 'Survived'], autopct='%1.1f%%')
    plt.xlabel('no_sibsp')
    plt.show()

def friends_survival(df):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    df[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
    ax[0].set_title('ParchandSurvived')
    df[['SibSp', 'Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
    ax[1].set_title('SibSpandSurvived')
    plt.show()

def fare_survival(df):
    plt.figure(figsize=(10, 5))
    df['Fare'].hist(bins=70)
    df.boxplot(column='Fare', by='Pclass', showfliers=False)
    plt.show()

def cabin_survival(df):
    df.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U0'
    df['Has_Cabin'] = df['Cabin'].apply(lambda x:0 if x == 'U0' else 1)
    df[['Has_Cabin', 'Survived']].groupby(['Has_Cabin']).mean().plot.bar()
    plt.show()

def fare_distribution(df):
    fare_not_survived = df['Fare'][df['Survived'] == 0]
    fare_survived = df['Fare'][df['Survived'] == 1]
    average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
    std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
    average_fare.plot(yerr=std_fare, kind='bar', legend=False)
    plt.show()

def cabin_class_survival(df):
    df['CabinLetter'] = df['Cabin'].map(lambda x:re.compile("([a-zA-Z]+)").search(x).group())
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
    df[['CabinLetter', 'Survived']].groupby(['CabinLetter']).mean().plot.bar()
    plt.show()

def embark_survival(df):
    sns.countplot('Embarked', hue='Survived', data=df)
    plt.title('EmbarkedandSurvived')
    sns.factorplot('Embarked', 'Survived', data=df, size=3, aspect=2)
    plt.title('EmbarkedandSurvivedrate')
    plt.show()

def dummy_var_embark(df):
    embark_dummies = pd.get_dummies(df['Embarked'])
    df = df.join(embark_dummies)
    df.drop(['Embarked'], axis=1, inplace=True)
    embark_dummies = df[['S', 'C', 'Q']]
    print (embark_dummies.head())

def cabin_var_factorizing(df):
    # Replacemissingvalueswith"U0"
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    # createfeatureforthealphabeticalpartofthecabinnumber
    df['CabinLetter'] = df['Cabin'].map(lambda x:re.compile("([a-zA-Z]+)").search(x).group())
    # convertthedistinctcabinletterswithincrementalintegervalues
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
    print (df['CabinLetter'].head())

def age_scaling(df):
    assert np.size(df['Age']) == 891
    # StandardScalerwillsubtractthemeanfromeachvaluethenscaletotheunitvariance
    scaler = preprocessing.StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))
    print (df['Age_scaled'].head())

def fare_binning(df):
    # Divideallfaresintoquartiles
    df['Fare_bin'] = pd.qcut(df['Fare'], 5)
    print (df['Fare_bin'].head())


if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')  # 训练数据集
    test_data = pd.read_csv('test.csv')  # 验证数据集
    sns.set_style('whitegrid')
    #print (train_data.head())  #显示数据
    #print (test_data.info())   #显示数据维度信息
    #titanic_plot_pie(train_data)  #生成训练集饼图
    #Age、Cabin、Embarked、Fare特征缺失值填充
    train_data = missing_value_process(train_data)  #完成缺失值填充
    train_data = missing_value_rf(train_data)   #基于随机森林完成缺失值填充
    #print (train_data.info())
    '''
    sex_survival_relation(train_data)   #性别与是否生存的关系
    pclass_survival_relation(train_data) #船舱等级和生存与否的关系
    pclass_sex_survival_relation(train_data)    #不同等级船舱的男女生存率
    age_survival_relation(train_data)   #年龄与存活与否的关系
    age_distribution(train_data)    #分析总体的年龄分布
    age_survival_facet(train_data)  #不同年龄下的生存和非生存的分布情况
    age_survival_prob(train_data)   #不同年龄下的平均生存率
    title_survival(train_data)  #观察不同称呼与生存率的关系
    name_length_survival(train_data)    #观察不同称呼与生存率的关系
    siblings_survival(train_data)   #观察兄弟姐妹与生存率的关系
    friends_survival(train_data)    #朋友和存活与否的关系
    fare_survival(train_data)   #票价分布和存活与否的关系
    fare_distribution(train_data)   #绘制生存与否与票价均值和方差的关系
    cabin_survival(train_data)  
    #cabin_class_survival(train_data)    #对不同类型的船舱进行分析
    embark_survival(train_data) #分析得出在不同的港口上船
    dummy_var_embark(train_data)    #生成哑变量
    cabin_var_factorizing(train_data)   #生成Factor
    age_scaling(train_data)
    fare_binning(train_data)
    '''
    #P.22










