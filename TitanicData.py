# -*- coding: utf-8 -*-
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
from xgboost import XGBClassifier
from sklearn.learning_curve import learning_curve


warnings.filterwarnings('ignore')


def titanic_plot_pie(df):
    #两个百分号应该是输出%自身，和C类似
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
    #df.values用列表输出dataframe
    #选取第一列外所有列
    X = age_df_notnull.values[:, 1:]
    #选取第一列数据
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
    sns.violinplot("Sex", "Age", hue="Survived", data=df, split=True, ax=ax[1])
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
    df.loc[df.Cabin.isnull(), 'Cabin'] = 'U0'
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
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    df['CabinLetter'] = df['Cabin'].map(lambda x:re.compile('([a-zA-Z]+)').search(x).group())
    #df['CabinLetter'] = list(map(lambda x: re.findall(r'([A-Z])', x)[0], df['Cabin']))
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
    df = missing_value_rf(df)
    # StandardScalerwillsubtractthemeanfromeachvaluethenscaletotheunitvariance
    # 归一化处理
    scaler = preprocessing.StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))
    print (df['Age_scaled'].head())

def fare_binning(df):
    # Divideallfaresintoquartiles
    df['Fare_bin'] = pd.qcut(df['Fare'], 5)
    print (df['Fare_bin'].head())
    # 在将数据分箱处理后，要么将数据factorize化，要么dummies化。
    # factorize
    df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0]
    # dummies
    #print (pd.get_dummies(df['Fare_bin']))
    fare_bin_dummies_df = pd.get_dummies(df['Fare_bin']).rename(columns=lambda x:'Fare_' + str(x))
    train_data = pd.concat([df, fare_bin_dummies_df], axis=1)
    #print (train_data)

def factoring():
    #在进行特征工程的时候，不仅需要对训练数据进行处理，还需要同时将测试数据同训练数据一起处理，使得二者具有相同的数据类型和数据分布。
    train_df_org = pd.read_csv('train.csv')
    test_df_org = pd.read_csv('test.csv')
    test_df_org['Survived'] = 0
    combined_train_test = train_df_org.append(test_df_org)
    PassengerId = test_df_org['PassengerId']

    # 对数据进行特征工程，也就是从各项参数中提取出对输出结果有或大或小的影响的特征，将这些特征作为训练模型的依据。一般来说，我们会先从含有缺失值的特征开始。
    # 因为“Embarked”字段的缺失值不多，所以这里我们以众数来填充：
    #print (combined_train_test['Embarked'].mode().iloc[0])
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
    # 为了后面的特征分析，这里我们将Embarked特征进行facrorizing
    combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
    # 使用pd.get_dummies获取one-hot编码
    emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'],\
                                    prefix=combined_train_test[['Embarked']].columns[0])
    combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)
    #print (combined_train_test)

    # 对Sex也进行one-hot编码，也就是dummy处理：
    # 为了后面的特征分析，这里我们也将Sex特征进行facrorizing
    combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
    sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
    combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

    # Name字段，首先先从名字中提取各种称呼：
    #combined_train_test['Title']=combined_train_test['Name'].map(lambda x:re.compile(",(. *?)\.").findall(x)[0])
    combined_train_test['Title'] = list(map(lambda x: re.findall(r', (.*?). ', x)[0], combined_train_test['Name']))
    # 将各式称呼进行统一化处理
    title_Dict = {}
    title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_Dict.update(dict.fromkeys(['Don', 'Sir', 'theCountess', 'Dona', 'Lady'], 'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
    combined_train_test['Title'] = combined_train_test['Title'].replace(title_Dict)
    #combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
    # 使用dummy对不同的称呼进行分列
    # 为了后面的特征分析，这里我们也将Title特征进行facrorizing
    combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
    title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
    combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)
    # 增加名字长度的特征：
    combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)
    # Fare字段，由前面分析可以知道，Fare项在测试数据中缺少一个值，所以需要对该值进行填充。
    # 我们按照一二三等舱各自的均价来填充：
    # 下面transform将函数np.mean应用到各个group中。
    combined_train_test['Fare']=combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
    # 通过对Ticket数据的分析，我们可以看到部分票号数据有重复，同时结合亲属人数及名字的数据，和票价船舱等级对比，
    # 我们可以知道购买的票中有家庭票和团体票，所以我们需要将团体票的票价分配到每个人的头上。
    combined_train_test['Group_Ticket']=combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
    combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
    combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)
    # 使用binning给票价分等级
    combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
    # 对于5个等级的票价我们也可以继续使用dummy为票价等级分列：
    combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
    fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns=lambda x:'Fare_' + str(x))
    combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_df], axis=1)
    combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)
    return combined_train_test


def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass']==1:
        if df['Fare']<=pclass1_mean_fare:
            return'Pclass1_Low'
        else:
            return'Pclass1_High'
    elif df['Pclass']==2:
        if df['Fare']<=pclass2_mean_fare:
            return'Pclass2_Low'
        else:
            return'Pclass2_High'
    elif df['Pclass']==3:
        if df['Fare']<=pclass3_mean_fare:
            return'Pclass3_Low'
        else:
            return'Pclass3_High'

def family_size_category(family_size):
    if family_size<=1:
        return 'Single'
    elif family_size<=4:
        return 'Small_Family'
    else:
        return 'Large_Family'


def Pclass_labeling(combined_train_test):
    # 建立PClassFareCategory
    Pclass1_mean_fare=combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]
    Pclass2_mean_fare=combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]
    Pclass3_mean_fare=combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]
    #print (Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare)
    # 建立Pclass_FareCategory
    combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
    #print (combined_train_test['Pclass_Fare_Category'])
    pclass_level = LabelEncoder()
    #print (pclass_level)
    # 给每一项添加标签
    pclass_level.fit(np.array(['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))
    # 转换成数值
    combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])
    # dummy转换
    pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(columns=lambda x: 'Pclass_' + str(x))
    combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)
    # 同时，我们将Pclass特征factorize化：
    combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]
    combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
    combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)
    le_family = LabelEncoder()
    le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
    combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])
    family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],
                                            prefix=combined_train_test[['Family_Size_Category']].columns[0])
    combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)
    return combined_train_test


def fill_missing_age(missing_age_train,missing_age_test):
    missing_age_X_train=missing_age_train.drop(['Age'],axis=1)
    missing_age_Y_train=missing_age_train['Age']
    missing_age_X_test=missing_age_test.drop(['Age'],axis=1)

    #gbm
    gbm_reg=GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid={'n_estimators':[2000],'max_depth':[4],'learning_rate':[0.01],'max_features':[3]}
    gbm_reg_grid=model_selection.GridSearchCV(gbm_reg,gbm_reg_param_grid,cv=10,n_jobs=25,verbose=1,scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train,missing_age_Y_train)
    print('AgefeatureBestGBParams:'+str(gbm_reg_grid.best_params_))
    print('AgefeatureBestGBScore:'+str(gbm_reg_grid.best_score_))
    print('GBTrainErrorfor"Age"FeatureRegressor:'+str(gbm_reg_grid.score(missing_age_X_train,missing_age_Y_train)))
    missing_age_test.loc[:,'Age_GB']=gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    # model2rf
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                               scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('AgefeatureBestRFParams:' + str(rf_reg_grid.best_params_))
    print('AgefeatureBestRFScore:' + str(rf_reg_grid.best_score_))
    print('RFTrainErrorfor"Age"FeatureRegressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])
    # two model smerge
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
    # missing_age_test['Age']=missing_age_test[['Age_GB','Age_LR']].mode(axis=1)
    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])
    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)
    return missing_age_test


def ticket_cabin_labeling(combined_train_test):
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter']\
                                            .apply(lambda x:'U0' if x.isnumeric() else x)
    # 如果要提取数字信息，则也可以这样做，现在我们对数字票单纯地分为一类。
    combined_train_test['Ticket_Number']=combined_train_test['Ticket'].apply(lambda x:\
                                        pd.to_numeric(x, errors='coerce'))
    #combined_train_test['Ticket_Number'].fillna(0,inplace=True)
    # 将Ticket_Letterfactorize
    combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]
    # Cabin字段，因为Cabin项的缺失值确实太多了，我们很难对其进行分析，或者预测。所以这里我
    #们可以直接将Cabin这一项特征去除。但通过上面的分析，可以知道，该特征信息的有无也与生存率
    #有一定的关系，所以这里我们暂时保留该特征，并将其分为有和无两类。
    combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
    combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x:0 if x == 'U0' else 1)
    return combined_train_test

def cor_plot(combined_train_test):
    # 特征间相关性分析
    # 我们挑选一些主要的特征，生成特征之间的关联图，查看特征与特征之间的相关性：
    Correlation = pd.DataFrame(combined_train_test[['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', \
                'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass','Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])
    colormap = plt.cm.viridis
    plt.figure(figsize=(14, 12))
    plt.title('PearsonCorrelationofFeatures', y=1.05, size=15)
    sns.heatmap(Correlation.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True,
                cmap=colormap,
                linecolor='white', annot=True)
    g = sns.pairplot(combined_train_test[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
                                          u'Family_Size', u'Title', u'Ticket_Letter']], hue='Survived',
                     palette='seismic', size=1.2, diag_kind=
                     'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
    g.set(xticklabels=[])
    plt.show()


def age_fare_regularing(combined_train_test):
    # 一些数据的正则化：这里我们将Age和fare进行正则化
    scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age', 'Fare', 'Name_length']])
    combined_train_test[['Age', 'Fare', 'Name_length']]=\
    scale_age_fare.transform(combined_train_test[['Age', 'Fare', 'Name_length']])
    #print (scale_age_fare)
    # 弃掉无用特征
    # 对于上面的特征工程中，我们从一些原始的特征中提取出了很多要融合到模型中的特征，但是我们
    #需要剔除那些原本的我们用不到的或者非数值特征：
    # 首先对我们的数据先进行一下备份，以便后期的再次分析：
    combined_data_backup = combined_train_test
    combined_train_test_temp=combined_train_test.drop(['Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category',
                              'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'], axis=1)
    combined_train_test.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category',
                              'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'], axis=1, inplace=True)
    return combined_train_test, combined_train_test_temp


def train_test_apart(combined_train_test):
    # 将训练数据和测试数据分开
    train_data = combined_train_test[:891]
    test_data = combined_train_test[891:]
    titanic_train_data_X = train_data.drop(['Survived'], axis=1)
    titanic_train_data_Y = train_data['Survived']
    titanic_test_data_X = test_data.drop(['Survived'], axis=1)
    #print (titanic_train_data_X.shape)
    return titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X


def get_top_n_features(titanic_train_data_X,titanic_train_data_Y,top_n_features):
    #randomforest
    rf_est=RandomForestClassifier(random_state=0)
    rf_param_grid={'n_estimators':[500],'min_samples_split':[2,3],'max_depth':[20]}
    rf_grid=model_selection.GridSearchCV(rf_est,rf_param_grid,n_jobs=25,cv=10,verbose=1)
    rf_grid.fit(titanic_train_data_X,titanic_train_data_Y)
    print('TopNFeaturesBestRFParams:'+str(rf_grid.best_params_))
    print('TopNFeaturesBestRFScore:'+str(rf_grid.best_score_))
    print('TopNFeaturesRFTrainScore:'+str(rf_grid.score(titanic_train_data_X,titanic_train_data_Y)))
    feature_imp_sorted_rf=pd.DataFrame({'feature':list(titanic_train_data_X),'importance':rf_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_rf=feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample10FeaturesfromRFClassifier')
    print(str(features_top_n_rf[:10]))

    #AdaBoost
    ada_est=AdaBoostClassifier(random_state=0)
    ada_param_grid={'n_estimators':[500],'learning_rate':[0.01,0.1]}
    ada_grid=model_selection.GridSearchCV(ada_est,ada_param_grid,n_jobs=25,cv=10,verbose=1)
    ada_grid.fit(titanic_train_data_X,titanic_train_data_Y)
    print('TopNFeaturesBestAdaParams:'+str(ada_grid.best_params_))
    print('TopNFeaturesBestAdaScore:'+str(ada_grid.best_score_))
    print('TopNFeaturesAdaTrainScore:'+str(ada_grid.score(titanic_train_data_X,titanic_train_data_Y)))
    feature_imp_sorted_ada=pd.DataFrame({'feature':list(titanic_train_data_X),
    'importance':
    ada_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_ada=feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample10FeaturefromAdaClassifier:')
    print(str(features_top_n_ada[:10]))

    #ExtraTree
    et_est=ExtraTreesClassifier(random_state=0)
    et_param_grid={'n_estimators':[500],'min_samples_split':[3,4],'max_depth':[20]}
    et_grid=model_selection.GridSearchCV(et_est,et_param_grid,n_jobs=25,cv=10,verbose=1)
    et_grid.fit(titanic_train_data_X,titanic_train_data_Y)
    print('TopNFeaturesBestETParams:'+str(et_grid.best_params_))
    print('TopNFeaturesBestETScore:'+str(et_grid.best_score_))
    print('TopNFeaturesETTrainScore:'+str(et_grid.score(titanic_train_data_X,titanic_train_data_Y)))
    feature_imp_sorted_et=pd.DataFrame({'feature':list(titanic_train_data_X),
    'importance':et_grid.best_estimator_.feature_importances_}).sort_values('importance',
    ascending=False)
    features_top_n_et=feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample10FeaturesfromETClassifier:')
    print(str(features_top_n_et[:10]))

    #GradientBoosting
    gb_est=GradientBoostingClassifier(random_state=0)
    gb_param_grid={'n_estimators':[500],'learning_rate':[0.01,0.1],'max_depth':[20]}
    gb_grid=model_selection.GridSearchCV(gb_est,gb_param_grid,n_jobs=25,cv=10,verbose=1)
    gb_grid.fit(titanic_train_data_X,titanic_train_data_Y)
    print('TopNFeaturesBestGBParams:'+str(gb_grid.best_params_))
    print('TopNFeaturesBestGBScore:'+str(gb_grid.best_score_))
    print('TopNFeaturesGBTrainScore:'+str(gb_grid.score(titanic_train_data_X,titanic_train_data_Y)))
    feature_imp_sorted_gb=pd.DataFrame({'feature':list(titanic_train_data_X),
    'importance':gb_grid.best_estimator_.feature_importances_}).sort_values('importance',
    ascending=False)
    features_top_n_gb=feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample10FeaturefromGBClassifier:')
    print(str(features_top_n_gb[:10]))

    #DecisionTree
    dt_est=DecisionTreeClassifier(random_state=0)
    dt_param_grid={'min_samples_split':[2,4],'max_depth':[20]}
    dt_grid=model_selection.GridSearchCV(dt_est,dt_param_grid,n_jobs=25,cv=10,verbose=1)
    dt_grid.fit(titanic_train_data_X,titanic_train_data_Y)
    print('TopNFeaturesBestDTParams:'+str(dt_grid.best_params_))
    print('TopNFeaturesBestDTScore:'+str(dt_grid.best_score_))
    print('TopNFeaturesDTTrainScore:'+str(dt_grid.score(titanic_train_data_X,titanic_train_data_Y)))
    feature_imp_sorted_dt=pd.DataFrame({'feature':list(titanic_train_data_X),
    'importance':dt_grid.best_estimator_.feature_importances_}).sort_values('importance',
    ascending=False)
    features_top_n_dt=feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample10FeaturesfromDTClassifier:')
    print(str(features_top_n_dt[:10]))

    #mergethethreemodels
    features_top_n=pd.concat([features_top_n_rf,features_top_n_ada,features_top_n_et,features_top_n_gb,features_top_n_dt],
    ignore_index=True).drop_duplicates()
    features_importance=pd.concat([feature_imp_sorted_rf,feature_imp_sorted_ada,feature_imp_sorted_et,
    feature_imp_sorted_gb,feature_imp_sorted_dt],ignore_index=True)
    return features_top_n, features_importance
    #return features_top_n,features_importance,feature_imp_sorted_rf,feature_imp_sorted_ada

def modeling(*args):
    titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X = args
    # 依据我们筛选出的特征构建训练集和测试集
    # 但如果在进行特征工程的过程中，产生了大量的特征，而特征与特征之间会存在一定的相关性。
    # 太多的特征一方面会影响模型训练的速度，另一方面也可能会使得模型过拟合。
    # 所以在特征太多的情况下，我们可以利用不同的模型对特征进行筛选，选取出我们想要的前n个特征。
    feature_to_pick = 30
    #feature_top_n, feature_importance, feature_imp_sorted_rf, feature_imp_sorted_ada = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
    feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
    #feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
    titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
    titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])
    #print ('*****************')
    #print (feature_importance)
    #print ('*****************')

    # 用视图可视化不同算法筛选的特征排序：
    rf_feature_imp = feature_importance[:10]
    #rf_feature_imp = feature_imp_sorted_rf[:10]
    #Ada_feature_imp = feature_importance[32:32 + 10].reset_index(drop=True)
    #为什么修改为68
    Ada_feature_imp = feature_importance[68:68 + 10].reset_index(drop=True)
    #Ada_feature_imp = feature_imp_sorted_ada[:10]
    # makeimportancesrelativetomaximportance
    rf_feature_importance = 100.0 * (rf_feature_imp['importance'] / rf_feature_imp['importance'].max())
    Ada_feature_importance = 100.0 * (Ada_feature_imp['importance']/Ada_feature_imp['importance'].max())
    # Gettheindexesofallfeaturesovertheimportancethreshold
    rf_important_idx = np.where(rf_feature_importance)[0]
    Ada_important_idx = [i for i in range(10)]
    #Ada_important_idx = np.where(Ada_feature_importance)[0]
    pos = np.arange(rf_important_idx.shape[0]) + .5
    plt.figure(1, figsize=(18, 8))
    plt.subplot(121)
    plt.barh(pos, rf_feature_importance[rf_important_idx][::-1])
    plt.yticks(pos, rf_feature_imp['feature'][::-1])
    plt.xlabel('RelativeImportance')
    plt.title('RandomForestFeatureImportance')
    plt.subplot(122)
    plt.barh(pos, Ada_feature_importance[Ada_important_idx][::-1])
    plt.yticks(pos, Ada_feature_imp['feature'][::-1])
    plt.xlabel('RelativeImportance')
    plt.title('AdaBoostFeatureImportance')
    plt.show()

'''
模型融合（ModelEnsemble）
常见的模型融合方法有：Bagging、Boosting、Stacking、Blending。
(6-1):Bagging
Bagging将多个模型，也就是多个基学习器的预测结果进行简单的加权平均或者投票。它的好处
是可以并行地训练基学习器。RandomForest就用到了Bagging的思想。
(6-2):Boosting
Boosting的思想，每个基学习器是在上一个基学习器学习的基础上，对上一个基学习器的错误进行
弥补。我们将会用到的AdaBoost，GradientBoost就用到了这种思想。
(6-3):Stacking
Stacking是用新的学习器去学习如何组合上一层的基学习器。如果把Bagging看作是多个基分类器
的线性组合，那么Stacking就是多个基分类器的非线性组合。Stacking可以将学习器一层一层地堆砌
起来，形成一个网状的结构。
相比来说Stacking的融合框架相对前面的二者来说在精度上确实有一定的提升，所以在下面的模
型融合上，我们也使用Stacking方法。
(6-4):Blending
Blending和Stacking很相似，但同时它可以防止信息泄露的问题。
Stacking框架融合:
这里我们使用了两层的模型融合：
Level1使用了：RandomForest、AdaBoost、ExtraTrees、GBDT、DecisionTree、KNN、SVM，一
共7个模型
Level2使用了XGBoost使用第一层预测的结果作为特征对最终的结果进行预测。
Level1：
Stacking框架是堆叠使用基础分类器的预测作为对二级模型的训练的输入。然而，我们不能简单
地在全部训练数据上训练基本模型，产生预测，输出用于第二层的训练。如果我们在TrainData上训
练，然后在TrainData上预测，就会造成标签。为了避免标签，我们需要对每个基学习器使用K-fold，
将K个模型对ValidSet的预测结果拼起来，作为下一层学习器的输入。
'''
def model_ensemble(*args):
    global combined_train_test_temp
    titanic_train_data_X, titanic_test_data_X, titanic_train_data_Y = args
    # 构建不同的基学习器，这里我们使用了RandomForest、AdaBoost、ExtraTrees、GBDT、DecisionTree、
    #KNN、SVM七个基学习器：（这里的模型可以使用如上面的GridSearch方法对模型的超参数进行搜
    #索选择）
    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt', max_depth=6,
                                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3,
                                    min_samples_leaf=2, max_depth=5, verbose=0)
    dt = DecisionTreeClassifier(max_depth=8)
    knn = KNeighborsClassifier(n_neighbors=2)
    svm = SVC(kernel='linear', C=0.025, max_iter=10000)
    #svm = SVC(kernel='linear', C=0.025)

    # 将pandas转换为arrays：
    # CreateNumpyarraysoftrain,testandtarget(Survived)dataframestofeedintoourmodels
    x_train = titanic_train_data_X.values  # Createsanarrayofthetraindata
    x_test = titanic_test_data_X.values  # Creatsanarrayofthetestdata
    y_train = titanic_train_data_Y.values
    ##CreateourOOFtrainandtestpredictions.Thesebaseresultswillbeusedasnewfeatures
    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test)  # RandomForest
    print ('rf_oof_train finished')
    ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test)  # AdaBoost
    print ('ada_oof_train finished')
    et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test)  # ExtraTrees
    print ('et_oof_train finished')
    gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test)  # GradientBoost
    print ('gb_oof_train finished')
    dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test)  # DecisionTree
    print ('dt_oof_train finished')
    knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test)  # KNeighbors
    print ('knn_oof_train finished')
    svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test)  # SupportVector
    print ('svm_oof_train finished')
    print("Training is complete")

    # 预测并生成提交文件
    # Level2：
    # 利用XGBoost，使用第一层预测的结果作为特征对最终的结果进行预测。
    x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train,
                              svm_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, \
                             svm_oof_test),axis=1)
    print ('xgbt data dealing')
    gbm = XGBClassifier(n_estimators=2000, max_depth=4, min_child_weight=2, gamma=0.9, subsample=0.8,
                        colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1).fit(x_train,y_train)
    print ('xgbt_train finished')
    predictions = gbm.predict(x_test)
    #处理PassengerId问题
    test_data_temp = combined_train_test_temp[891:]
    titanic_test_data_X_temp = test_data_temp.drop(['Survived'], axis=1)
    #####################################################################
    StackingSubmission = pd.DataFrame({'PassengerId': titanic_test_data_X_temp['PassengerId'], 'Survived': predictions})
    StackingSubmission.to_csv('StackingSubmission.csv', index=False, sep=',')

#L1:这里我们建立输出fold预测方法
def get_out_fold(clf,x_train,y_train,x_test):
    global ntrain, ntest, SEED, NFOLDS, kf
    ntrain = titanic_train_data_X.shape[0]
    ntest = titanic_test_data_X.shape[0]
    SEED = 0  # forreproducibility
    NFOLDS = 7  # setfoldsforout-of-foldprediction
    kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)
    oof_train=np.zeros((ntrain,))
    oof_test=np.zeros((ntest,))
    oof_test_skf=np.empty((NFOLDS,ntest))
    for i,(train_index,test_index) in enumerate(kf.split(x_train)):
        x_tr=x_train[train_index]
        y_tr=y_train[train_index]
        x_te=x_train[test_index]
        clf.fit(x_tr,y_tr)
        oof_train[test_index]=clf.predict(x_te)
        oof_test_skf[i,:]=clf.predict(x_test)
    oof_test[:]=oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)


def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.1,1.0,5),verbose=0):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Trainingexamples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Trainingscore")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validationscore")
    plt.legend(loc="best")
    return plt


def model_running(titanic_train_data_X, titanic_train_data_Y):
    x_train = titanic_train_data_X.values  # Createsanarrayofthetraindata
    y_train = titanic_train_data_Y.values
    X = x_train
    Y = y_train
    # RandomForest
    rf_parameters = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2,
                     'max_features': 'sqrt', 'verbose': 0}
    # AdaBoost
    ada_parameters = {'n_estimators': 500, 'learning_rate': 0.1}
    # ExtraTrees
    et_parameters = {'n_jobs': -1, 'n_estimators': 500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}
    # GradientBoosting
    gb_parameters = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}
    # DecisionTree
    dt_parameters = {'max_depth': 8}
    # KNeighbors
    knn_parameters = {'n_neighbors': 2}
    # SVM
    svm_parameters = {'kernel': 'linear', 'C': 0.025}
    # XGB
    #gbm_parameters = {'n_estimators': 2000, 'max_depth': 4, 'min_child_weight': 2, 'gamma': 0.9, 'subsample': 0.8,
    #                  'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'nthread': -1, 'scale_pos_weight': 1}
    title = "LearningCurves"
    plot_learning_curve(RandomForestClassifier(**rf_parameters), title, X, Y, cv=None, n_jobs=4, train_sizes=[50, 100,
                                    150, 200, 250, 350, 400, 450, 500])
    plt.show()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
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

    #数据Plot分析
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
    #数据预处理分析
    '''
    combined_train_test = factoring()
    combined_train_test = Pclass_labeling(combined_train_test)
    # Age字段，因为Age项的缺失值较多，所以不能直接填充age的众数或者平均数。
    # 常见的有两种对年龄的填充方式：一种是根据Title中的称呼，如Mr，Master、Miss等称呼不同类
    # 别的人的平均年龄来填充；一种是综合几项如Sex、Title、Pclass等其他没有缺失值的项，使用机器学习算法来预测Age。
    # 这里我们使用后者来处理。以Age为目标值，将Age完整的项作为训练集，将Age缺失的项作为测试集。
    missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size',
                                                       'Family_Size_Category', 'Fare', 'Fare_bin_id', 'Pclass']])
    missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
    missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
    # print (missing_age_test.head())
    combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train,missing_age_test)


    combined_train_test.to_csv('combined_train_test.csv')
    '''

    #模型构建
    '''
    combined_train_test = pd.read_csv('combined_train_test.csv')
    lst = combined_train_test.columns
    lst = lst[1:]
    combined_train_test = combined_train_test[lst]

    combined_train_test = ticket_cabin_labeling(combined_train_test)
    #cor_plot(combined_train_test)

    combined_train_test, combined_train_test_temp = age_fare_regularing(combined_train_test)
    combined_train_test['Ticket_Number'].fillna(0, inplace=True)
    #combined_train_test.to_csv('combined_train_test_2.csv')

    #dropna()
    #combined_train_test = combined_train_test.dropna()
    titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X = train_test_apart(combined_train_test)

    fields = ['Age', 'Fare', 'Ticket_Number']
    titanic_train_data_X[fields] = titanic_train_data_X[fields].astype(float)
    titanic_test_data_X[fields] = titanic_test_data_X[fields].astype(float)
    modeling(titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X)
    model_ensemble(titanic_train_data_X, titanic_test_data_X, titanic_train_data_Y)

    #模型图
    model_running(titanic_train_data_X, titanic_train_data_Y)
    '''








