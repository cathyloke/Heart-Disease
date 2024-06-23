#importing the libraries
import pandas as pd 
import numpy as np

#matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier


#reading the dataset
df = pd.read_csv('heart_2020_cleaned.csv')
#return first 5 rows in the dataset
df.head()

#summary of numerical features
df.describe().T.style.set_properties(**{'background-color': 'grey','color': 'white','border-color': 'white'})

#brief information
print('\nThe columns are:  ')
print(df.columns)

print('\n\nNumber of X tuples = {}'.format(df.shape[0]))

#print the information about the dataframe which is the column name, number, number of null value, data type and the memory usage
print('\n\n')
print(df.info())
print('\n\n')
print(df.head())

#return the number of unique values for each column
print('\n\n')
print('The unique values for each column : ')
print(df.nunique())
    
#return the number of missing values for each column
print('\n\n')
print('The number of missing values for each column : ')
df_columns = df.columns.tolist()
for col in df_columns:
    print('{} column missing values: {}'.format(col, df[col].isnull().sum()))



#Calculate the statistics measure and plot the boxplot
def statistics():
    #Statistics measure calculation for BMI, Physical Health, Mental Health and Sleep Time
    # Calculate bmi statistics for all
    bmi_mode = df['BMI'].mode()[0]
    bmi_mean = df['BMI'].mean()
    bmi_max = df['BMI'].max()
    bmi_min = df['BMI'].min()
    bmi_std = df['BMI'].std()
    bmi_median = df['BMI'].median()
    stats = df['BMI'].describe()
    bmi_q1 = stats['25%']
    bmi_q3 = stats['75%']
    bmi_iqr = bmi_q3 - bmi_q1
    upper_bound = bmi_q3 + 1.5*bmi_iqr
    lower_bound = bmi_q1 - 1.5*bmi_iqr
    bmi_outliers = df[(df['BMI'] < lower_bound) | (df['BMI'] > upper_bound)]

    # Calculate BMI statistics for individuals with heart disease
    bmi_mode_hd = df[df['HeartDisease'] == "Yes"]['BMI'].mode()[0]
    bmi_mean_hd = df[df['HeartDisease'] == "Yes"]['BMI'].mean()
    bmi_max_hd = df[df['HeartDisease'] == "Yes"]['BMI'].max()
    bmi_min_hd = df[df['HeartDisease'] == "Yes"]['BMI'].min()
    bmi_std_hd = df[df['HeartDisease'] == "Yes"]['BMI'].std()
    bmi_median_hd = df[df['HeartDisease'] == "Yes"]['BMI'].median()
    stats_hd = df[df['HeartDisease'] == "Yes"]['BMI'].describe()
    bmi_q1_hd = stats_hd['25%']
    bmi_q3_hd = stats_hd['75%']
    bmi_iqr_hd = bmi_q3_hd - bmi_q1_hd
    upper_bound_hd = bmi_q3_hd + 1.5*bmi_iqr_hd
    lower_bound_hd = bmi_q1_hd - 1.5*bmi_iqr_hd
    bmi_outliers_hd = df[(df['BMI'] < lower_bound_hd) | (df['BMI'] > upper_bound_hd) & (df['HeartDisease'] == 1)]

    # Calculate BMI statistics for individuals without heart disease
    bmi_mode_no_hd = df[df['HeartDisease'] == "Yes"]['BMI'].mode()[0]
    bmi_mean_no_hd = df[df['HeartDisease'] == "No"]['BMI'].mean()
    bmi_max_no_hd = df[df['HeartDisease'] == "No"]['BMI'].max()
    bmi_min_no_hd = df[df['HeartDisease'] == "No"]['BMI'].min()
    bmi_std_no_hd = df[df['HeartDisease'] == "No"]['BMI'].std()
    bmi_median_no_hd = df[df['HeartDisease'] == "No"]['BMI'].median()
    stats_no_hd = df[df['HeartDisease'] == "No"]['BMI'].describe()
    bmi_q1_no_hd = stats_no_hd['25%']
    bmi_q3_no_hd = stats_no_hd['75%']
    bmi_iqr_no_hd = bmi_q3_no_hd - bmi_q1_no_hd
    upper_bound_no_hd = bmi_q3_no_hd + 1.5*bmi_iqr_no_hd
    lower_bound_no_hd = bmi_q1_no_hd - 1.5*bmi_iqr_no_hd
    bmi_outliers_no_hd = df[(df['BMI'] < lower_bound_no_hd) | (df['BMI'] > upper_bound_no_hd) & (df['HeartDisease'] == 0)]

    # Calculate physical health statistics for all 
    ph_mode = df['PhysicalHealth'].mode()[0]
    ph_mean = df['PhysicalHealth'].mean()
    ph_max = df['PhysicalHealth'].max()
    ph_min = df['PhysicalHealth'].min()
    ph_std = df['PhysicalHealth'].std()
    ph_median = df['PhysicalHealth'].median()
    stats = df['PhysicalHealth'].describe()
    ph_q1 = stats['25%']
    ph_q3 = stats['75%']
    ph_iqr = ph_q3 - ph_q1
    upper_bound = ph_q3 + 1.5*ph_iqr
    lower_bound = ph_q1 - 1.5*ph_iqr
    ph_outliers = df[(df['PhysicalHealth'] < lower_bound) | (df['PhysicalHealth'] > upper_bound)]

    # Calculate physical health statistics for individuals with heart disease
    ph_mode_hd = df[df['HeartDisease'] == "Yes"]['PhysicalHealth'].mode()[0]
    ph_mean_hd = df[df['HeartDisease'] == "Yes"]['PhysicalHealth'].mean()
    ph_max_hd = df[df['HeartDisease'] == "Yes"]['PhysicalHealth'].max()
    ph_min_hd = df[df['HeartDisease'] == "Yes"]['PhysicalHealth'].min()
    ph_std_hd = df[df['HeartDisease'] == "Yes"]['PhysicalHealth'].std()
    ph_median_hd = df[df['HeartDisease'] == "Yes"]['PhysicalHealth'].median()
    stats_hd = df[df['HeartDisease'] == "Yes"]['PhysicalHealth'].describe()
    ph_q1_hd = stats_hd['25%']
    ph_q3_hd = stats_hd['75%']
    ph_iqr_hd = ph_q3_hd - ph_q1_hd
    upper_bound_hd = ph_q3_hd + 1.5*ph_iqr_hd
    lower_bound_hd = ph_q1_hd - 1.5*ph_iqr_hd
    ph_outliers_hd = df[(df['PhysicalHealth'] < lower_bound_hd) | (df['PhysicalHealth'] > upper_bound_hd) & (df['HeartDisease'] == 1)]

    # Calculate physical health statistics for individuals without heart disease
    ph_mode_no_hd = df[df['HeartDisease'] == "No"]['PhysicalHealth'].mode()[0]
    ph_mean_no_hd = df[df['HeartDisease'] == "No"]['PhysicalHealth'].mean()
    ph_max_no_hd = df[df['HeartDisease'] == "No"]['PhysicalHealth'].max()
    ph_min_no_hd = df[df['HeartDisease'] == "No"]['PhysicalHealth'].min()
    ph_std_no_hd = df[df['HeartDisease'] == "No"]['PhysicalHealth'].std()
    ph_median_no_hd = df[df['HeartDisease'] == "No"]['PhysicalHealth'].median()
    stats_no_hd = df[df['HeartDisease'] == "No"]['PhysicalHealth'].describe()
    ph_q1_no_hd = stats_no_hd['25%']
    ph_q3_no_hd = stats_no_hd['75%']
    ph_iqr_no_hd = ph_q3_no_hd - ph_q1_no_hd
    upper_bound_no_hd = ph_q3_no_hd + 1.5*ph_iqr_no_hd
    lower_bound_no_hd = ph_q1_no_hd - 1.5*ph_iqr_no_hd
    ph_outliers_no_hd = df[(df['PhysicalHealth'] < lower_bound_no_hd) | (df['PhysicalHealth'] > upper_bound_no_hd) & (df['HeartDisease'] == 0)]

    # Calculate mental health statistics for all
    mh_mode = df['MentalHealth'].mode()[0]
    mh_mean = df['MentalHealth'].mean()
    mh_max = df['MentalHealth'].max()
    mh_min = df['MentalHealth'].min()
    mh_std = df['MentalHealth'].std()
    mh_median = df['MentalHealth'].median()
    stats = df['MentalHealth'].describe()
    mh_q1 = stats['25%']
    mh_q3 = stats['75%']
    mh_iqr = mh_q3 - mh_q1
    upper_bound = mh_q3 + 1.5*mh_iqr
    lower_bound = mh_q1 - 1.5*mh_iqr
    mh_outliers = df[(df['MentalHealth'] < lower_bound) | (df['MentalHealth'] > upper_bound)]

    # Calculate mental health statistics for individuals with heart disease
    mh_mode_hd = df[df['HeartDisease'] == "Yes"]['MentalHealth'].mode()[0]
    mh_mean_hd = df[df['HeartDisease'] == "Yes"]['MentalHealth'].mean()
    mh_max_hd = df[df['HeartDisease'] == "Yes"]['MentalHealth'].max()
    mh_min_hd = df[df['HeartDisease'] == "Yes"]['MentalHealth'].min()
    mh_std_hd = df[df['HeartDisease'] == "Yes"]['MentalHealth'].std()
    mh_median_hd = df[df['HeartDisease'] == "Yes"]['MentalHealth'].median()
    stats_hd = df[df['HeartDisease'] == "Yes"]['MentalHealth'].describe()
    mh_q1_hd = stats_hd['25%']
    mh_q3_hd = stats_hd['75%']
    mh_iqr_hd = mh_q3_hd - mh_q1_hd
    upper_bound_hd = mh_q3_hd + 1.5*mh_iqr_hd
    lower_bound_hd = mh_q1_hd - 1.5*mh_iqr_hd
    mh_outliers_hd = df[(df['MentalHealth'] < lower_bound_hd) | (df['MentalHealth'] > upper_bound_hd) & (df['HeartDisease'] == 1)]

    # Calculate mental health statistics for individuals without heart disease
    mh_mode_no_hd = df[df['HeartDisease'] == "No"]['MentalHealth'].mode()[0]
    mh_mean_no_hd = df[df['HeartDisease'] == "No"]['MentalHealth'].mean()
    mh_max_no_hd = df[df['HeartDisease'] == "No"]['MentalHealth'].max()
    mh_min_no_hd = df[df['HeartDisease'] == "No"]['MentalHealth'].min()
    mh_std_no_hd = df[df['HeartDisease'] == "No"]['MentalHealth'].std()
    mh_median_no_hd = df[df['HeartDisease'] == "No"]['MentalHealth'].median()
    stats_no_hd = df[df['HeartDisease'] == "No"]['MentalHealth'].describe()
    mh_q1_no_hd = stats_no_hd['25%']
    mh_q3_no_hd = stats_no_hd['75%']
    mh_iqr_no_hd = mh_q3_no_hd - mh_q1_no_hd
    upper_bound_no_hd = mh_q3_no_hd + 1.5*mh_iqr_no_hd
    lower_bound_no_hd = mh_q1_no_hd - 1.5*mh_iqr_no_hd
    mh_outliers_no_hd = df[(df['MentalHealth'] < lower_bound_no_hd) | (df['MentalHealth'] > upper_bound_no_hd) & (df['HeartDisease'] == 0)]

    # Calculate sleep time statistics for all 
    st_mode = df['SleepTime'].mode()[0]
    st_mean = df['SleepTime'].mean()
    st_max = df['SleepTime'].max()
    st_min = df['SleepTime'].min()
    st_std = df['SleepTime'].std()
    st_median = df['SleepTime'].median()
    stats = df['SleepTime'].describe()
    st_q1 = stats['25%']
    st_q3 = stats['75%']
    st_iqr = st_q3 - st_q1
    upper_bound = st_q3 + 1.5*st_iqr
    lower_bound = st_q1 - 1.5*st_iqr
    st_outliers = df[(df['SleepTime'] < lower_bound) | (df['SleepTime'] > upper_bound)]


    # Calculate sleep time statistics for individuals with heart disease
    st_mode_hd = df[df['HeartDisease'] == "Yes"]['SleepTime'].mode()[0]
    st_mean_hd = df[df['HeartDisease'] == "Yes"]['SleepTime'].mean()
    st_max_hd = df[df['HeartDisease'] == "Yes"]['SleepTime'].max()
    st_min_hd = df[df['HeartDisease'] == "Yes"]['SleepTime'].min()
    st_std_hd = df[df['HeartDisease'] == "Yes"]['SleepTime'].std()
    st_median_hd = df[df['HeartDisease'] == "Yes"]['SleepTime'].median()
    stats_hd = df[df['HeartDisease'] == "Yes"]['SleepTime'].describe()
    st_q1_hd = stats_hd['25%']
    st_q3_hd = stats_hd['75%']
    st_iqr_hd = st_q3_hd - st_q1_hd
    upper_bound_hd = st_q3_hd + 1.5*st_iqr_hd
    lower_bound_hd = st_q1_hd - 1.5*st_iqr_hd
    st_outliers_hd = df[(df['SleepTime'] < lower_bound_hd) | (df['SleepTime'] > upper_bound_hd) & (df['HeartDisease'] == 1)]

    # Calculate sleep time statistics for individuals without heart disease
    st_mode_no_hd = df[df['HeartDisease'] == "No"]['SleepTime'].mode()[0]
    st_mean_no_hd = df[df['HeartDisease'] == "No"]['SleepTime'].mean()
    st_max_no_hd = df[df['HeartDisease'] == "No"]['SleepTime'].max()
    st_min_no_hd = df[df['HeartDisease'] == "No"]['SleepTime'].min()
    st_std_no_hd = df[df['HeartDisease'] == "No"]['SleepTime'].std()
    st_median_no_hd = df[df['HeartDisease'] == "No"]['SleepTime'].median()
    stats_no_hd = df[df['HeartDisease'] == "No"]['SleepTime'].describe()
    st_q1_no_hd = stats_no_hd['25%']
    st_q3_no_hd = stats_no_hd['75%']
    st_iqr_no_hd = st_q3_no_hd - st_q1_no_hd
    upper_bound_no_hd = st_q3_no_hd + 1.5*st_iqr_no_hd
    lower_bound_no_hd = st_q1_no_hd - 1.5*st_iqr_no_hd
    st_outliers_no_hd = df[(df['SleepTime'] < lower_bound_no_hd) | (df['SleepTime'] > upper_bound_no_hd) & (df['HeartDisease'] == 0)]


    #print the result
    print("\n\n")
    print("The statistics measure for BMI, Physical Health, Mental Health and Sleep Time")
    print("BMI: \nmode = {:.2f}, \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(bmi_mode), bmi_mean, bmi_median, bmi_max, bmi_min, bmi_std))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(bmi_q1,bmi_q3,bmi_iqr))
    print("\nPhysicalHealth: \nmode = {:.2f} \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(ph_mode), ph_mean, ph_median, ph_max, ph_min, ph_std))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(ph_q1,ph_q3,ph_iqr))
    print("\nMentalHealth: \nmode = {:.2f} \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(mh_mode), mh_mean, mh_median, mh_max, mh_min, mh_std))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(mh_q1,mh_q3,mh_iqr))
    print("\nSleepTime: \nmode = {:.2f} \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(st_mode), st_mean, st_median, st_max, st_min, st_std))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(st_q1,st_q3,st_iqr))

    print("\n\nThe statistics for BMI, Physical Health, Mental Health and Sleep Time in terms of heart disease")
    print("BMI: \nmode = {:.2f}, \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(bmi_mode_hd), bmi_mean_hd, bmi_median_hd, bmi_max_hd, bmi_min_hd, bmi_std_hd))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(bmi_q1_hd,bmi_q3_hd,bmi_iqr_hd))
    print("\nPhysicalHealth: \nmode = {:.2f}, \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(ph_mode_hd), ph_mean_hd, ph_median_hd, ph_max_hd, ph_min_hd, ph_std_hd))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(ph_q1_hd,ph_q3_hd,ph_iqr_hd))
    print("\nMentalHealth: \nmode = {:.2f}, \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(mh_mode_hd), mh_mean_hd, mh_median_hd, mh_max_hd, mh_min_hd, mh_std_hd))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(mh_q1_hd,mh_q3_hd,mh_iqr_hd))
    print("\nSleepTime: \nmode = {:.2f}, \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(st_mode_hd), st_mean_hd, st_median_hd, st_max_hd, st_min_hd, st_std_hd))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(st_q1_hd,st_q3_hd,st_iqr_hd))

    print("\n\nThe statistics for BMI, Physical Health, Mental Health and Sleep Time in terms of normal")
    print("BMI: \nmode = {:.2f}, \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(bmi_mode_no_hd), bmi_mean_no_hd, bmi_median_no_hd, bmi_max_no_hd, bmi_min_no_hd, bmi_std_no_hd))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(bmi_q1_no_hd,bmi_q3_no_hd,bmi_iqr_no_hd))
    print("\nPhysicalHealth: \nmode = {:.2f}, \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(ph_mode_no_hd), ph_mean_no_hd, ph_median_no_hd, ph_max_no_hd, ph_min_no_hd, ph_std_no_hd))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(ph_q1_no_hd,ph_q3_no_hd,ph_iqr_no_hd))
    print("\nMentalHealth: \nmode = {:.2f}, \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(mh_mode_no_hd), mh_mean_no_hd, mh_median_no_hd, mh_max_no_hd, mh_min_no_hd, mh_std_no_hd))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(mh_q1_no_hd,mh_q3_no_hd,mh_iqr_no_hd))
    print("\nSleepTime: \nmode = {:.2f}, \nmean = {:.2f}, \nmedian = {:.2f}, \nmax = {:.2f}, \nmin = {:.2f}, \nstd = {:.2f}".format(float(st_mode_no_hd), st_mean_no_hd, st_median_no_hd, st_max_no_hd, st_min_no_hd, st_std_no_hd))
    print("Quartile 1: {:.2f}, \nQuartile 3 = {:.2f}, \nInterquartile range = {:.2f}".format(st_q1_no_hd,st_q3_no_hd,st_iqr_no_hd))


    
    #Identify outliers for BMI, Physical Health, Mental Health and Sleep Time
    print("\nThe outliers for BMI, Physical Health, Mental Health and Sleep Time")
    #Outliers for BMI
    print("Outlier of the BMI")
    print('Outliers:')
    print(bmi_outliers)
    #Count the heart diseases case based on BMI
    heart_disease_outliers = bmi_outliers[bmi_outliers['HeartDisease'] == "Yes"]
    count = heart_disease_outliers.shape[0]
    print('Number of heart disease cases related to BMI outliers:', count)
    heart_disease_outliers = bmi_outliers[bmi_outliers['HeartDisease'] == "No"]
    count = heart_disease_outliers.shape[0]
    print('Number of no heart disease cases related to BMI outliers:', count)

    #Outliers for Physical Health
    print("\n\nOutlier of the Physical Health")
    print('Outliers:')
    print(ph_outliers)
    #Count the heart diseases case based on BMI
    heart_disease_outliers = ph_outliers[ph_outliers['HeartDisease'] == "Yes"]
    count = heart_disease_outliers.shape[0]
    print('Number of heart disease cases related to Physical Health outliers:', count)
    heart_disease_outliers = ph_outliers[ph_outliers['HeartDisease'] == "No"]
    count = heart_disease_outliers.shape[0]
    print('Number of no heart disease cases related to Physical Health outliers:', count)

    #Outliers for Mental Health
    print("\n\nOutlier of the Mental Health")
    print('Outliers:')
    print(mh_outliers)
    #Count the heart diseases case based on BMI
    heart_disease_outliers = mh_outliers[mh_outliers['HeartDisease'] == "Yes"]
    count = heart_disease_outliers.shape[0]
    print('Number of heart disease cases related to Mental Health outliers:', count)
    heart_disease_outliers = mh_outliers[mh_outliers['HeartDisease'] == "No"]
    count = heart_disease_outliers.shape[0]
    print('Number of no heart disease cases related to Mental Health outliers:', count)

    #Outliers for Sleep Time
    print("\n\nOutlier of the Sleep Time")
    print('Outliers:')
    print(st_outliers)
    #Count the heart diseases case based on BMI
    heart_disease_outliers = st_outliers[st_outliers['HeartDisease'] == "Yes"]
    count = heart_disease_outliers.shape[0]
    print('Number of heart disease cases related to Sleep Time outliers:', count)
    heart_disease_outliers = st_outliers[st_outliers['HeartDisease'] == "No"]
    count = heart_disease_outliers.shape[0]
    print('Number of no heart disease cases related to Sleep Time outliers:', count)

    cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime'] 
    #Plot the boxplot    
    sns.boxplot(data=df[cols])
    plt.title('Distribution of BMI, Physical and Mental Health Scores, SleepTime')
    plt.xlabel('Variable')
    plt.ylabel('Score')
    plt.show()

    #Plot the boxplot
    sns.boxplot(data=df[df['HeartDisease'] == "Yes"][cols])
    plt.title('Distribution of BMI, Physical and Mental Health Scores, SleepTime for heart disease')
    plt.xlabel('Variable')
    plt.ylabel('Score')
    plt.show()

    #Plot the boxplot
    sns.boxplot(data=df[df['HeartDisease'] == "No"][cols])
    plt.title('Distribution of BMI, Physical and Mental Health Scores, SleepTime for normal')
    plt.xlabel('Variable')
    plt.ylabel('Score')
    plt.show()

statistics()

#visualization of categorical features
def graph():
    #exploratory analysis
    #   visualization of categorical features
    #create a histogram plot that shows the distribution of with and without heart disease based on age category
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x='AgeCategory', hue='HeartDisease', data=df, palette='YlOrBr', order=sorted(df['AgeCategory'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease according to AgeCategory")
    plt.xlabel('AgeCategory')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on sex
    plt.figure(figsize = (13,6))
    ax = sns.countplot( x= 'Sex', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['Sex'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])    
    plt.suptitle("Distribution of Cases with Yes/No heartdisease according to sex")
    plt.xlabel('Sex')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Count')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on race
    plt.figure(figsize = (13,6))
    ax = sns.countplot( x= 'Race', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['Race'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease according to race")
    plt.xlabel('Race')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on smoker
    plt.figure(figsize = (13,6))
    ax = sns.countplot( x= 'Smoking', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['Smoking'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease according to smoker or not")
    plt.xlabel('Smoking')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Count') 
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on Alcohol Drinking
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x = 'AlcoholDrinking', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['AlcoholDrinking'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease based on Alcohol Drinking")
    plt.xlabel('AlcoholDrinking')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on Physical Activity
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x = 'PhysicalActivity', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['PhysicalActivity'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease based on Physical Activity")
    plt.xlabel('PhysicalActivity')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on General Health
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x = 'GenHealth', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease based on General Health")
    plt.xlabel('GenHealth')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on Difficult Walking
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x = 'DiffWalking', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['DiffWalking'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease based on Difficult Walking")
    plt.xlabel('DiffWalking')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on stroke
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x = 'Stroke', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['Stroke'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease based on previous exposure to Stroke")
    plt.xlabel('Stroke')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()
    
    #create a histogram plot that shows the distribution of with and without heart disease based on Asthma
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x = 'Asthma', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['Asthma'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease based on Asthma")
    plt.xlabel('Asthma')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on diabetic
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x = 'Diabetic', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['Diabetic'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease based on previous exposure to Diabetic")
    plt.xlabel('Diabetic')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on kidney disease
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x = 'KidneyDisease', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['KidneyDisease'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease according to kidneydisease")
    plt.xlabel('KidneyDisease')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

    #create a histogram plot that shows the distribution of with and without heart disease based on skin cancer
    plt.figure(figsize = (13,6))
    ax = sns.countplot(x = 'SkinCancer', hue = 'HeartDisease', data = df, palette = 'YlOrBr', order=sorted(df['SkinCancer'].unique()))
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    plt.suptitle("Distribution of Cases with Yes/No heartdisease based on previous exposure to skin cancer")
    plt.xlabel('SkinCancer')
    plt.legend(['Normal','HeartDisease'])
    plt.ylabel('Frequency')
    plt.show()

#graph()


#visualization of numerical features - BMI, SleepTime, PhysicalHealth, MentalHealth
#Replace the value Yes, Male as 1 and No, Female as 0
df =  df[df.columns].replace({'No, borderline diabetes':'No','Yes (during pregnancy)':'Yes' })
df =  df[df.columns].replace({'Yes':1, 'No':0, 'Male':1,'Female':0})

#graph plot of heat map & Kernel Distribution Estimation Plot
def correlation():
    sns.set_style('white')
    sns.set_palette('YlOrBr')

    #heatmap - correlation among all the variables
    correlation = df.corr(numeric_only=True).round(2)
    plt.figure(figsize = (14,7))
    sns.heatmap(correlation, annot = True, cmap = 'YlOrBr')
    plt.xticks(rotation=30, fontsize=10)
    plt.show()

    #calculate the absolute correlation values between all features and the "HeartDisease" target variable
    #bar chart - distribution of correlation of all variables
    plt.figure(figsize = (13,6))
    plt.title('Distribution of correlation of features')
    abs(correlation['HeartDisease']).sort_values()[:-1].plot.barh()
    plt.show()

    #Kernel Distribution Estimation Plot - disribution of BMI
    fig, ax = plt.subplots(figsize = (13,5))
    sns.kdeplot(df[df["HeartDisease"]==1]["BMI"], alpha=0.5,fill = True, color="red", label="HeartDisease", ax = ax)
    sns.kdeplot(df[df["HeartDisease"]==0]["BMI"], alpha=0.5,fill = True, color="#fccc79", label="Normal", ax = ax)
    plt.title('Kernel Distribution Estimation Plot of Body Mass Index', fontsize = 18)
    ax.set_xlabel("BodyMass")
    ax.set_ylabel("Frequency")
    ax.legend();
    plt.show()

    #Kernel Distribution Estimation Plot - distribution based on sleep time
    fig, ax = plt.subplots(figsize = (13,5))
    sns.kdeplot(df[df["HeartDisease"]==1]["SleepTime"], alpha=0.5,fill = True, color="red", label="HeartDisease", ax = ax)
    sns.kdeplot(df[df["HeartDisease"]==0]["SleepTime"], alpha=0.5,fill = True, color="#fccc79", label="Normal", ax = ax)
    plt.title('Kernel Distribution Estimation Plot of SleepTime values', fontsize = 18)
    ax.set_xlabel("SleepTime")
    ax.set_ylabel("Frequency")
    ax.legend();
    plt.show()

    #Kernel Distribution Estimation Plot - distribution based on physical health
    fig, ax = plt.subplots(figsize = (13,5))
    sns.kdeplot(df[df["HeartDisease"]==1]["PhysicalHealth"], alpha=0.5,fill = True, color="red", label="HeartDisease", ax = ax)
    sns.kdeplot(df[df["HeartDisease"]==0]["PhysicalHealth"], alpha=0.5,fill = True, color="#fccc79", label="Normal", ax = ax)
    plt.title('Kernel Distribution Estimation Plot of PhysicalHealth state for the last 30 days', fontsize = 18) # Read the introduction to know what the scale of numerical features mean
    ax.set_xlabel("PhysicalHealth")
    ax.set_ylabel("Frequency")
    ax.legend();
    plt.show()

    #Kernel Distribution Estimation Plot - distribution based on mental health
    fig, ax = plt.subplots(figsize = (13,5))
    sns.kdeplot(df[df["HeartDisease"]==1]["MentalHealth"], alpha=0.5,fill = True, color="red", label="HeartDisease", ax = ax)
    sns.kdeplot(df[df["HeartDisease"]==0]["MentalHealth"], alpha=0.5,fill = True, color="#fccc79", label="Normal", ax = ax)
    plt.title('Kernel Distribution Estimation Plot of MentalHealth state for the last 30 days', fontsize = 18)
    ax.set_xlabel("MentalHealth")
    ax.set_ylabel("Frequency")
    ax.legend();
    plt.show()
    
correlation()


#to avoid imbalance data
def oversampling():
    df = pd.read_csv('heart_2020_cleaned.csv')
    # split dataset into features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # perform oversampling using RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Add the 'y' column back to the oversampled data
    Xy_oversampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
    Xy_oversampled.columns = df.columns

    # Save the oversampled data to a new CSV file
    Xy_oversampled.to_csv('oversampled_data.csv', index=False)

    # plot class distribution before and after oversampling
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].bar(['0', '1'], y.value_counts().values)
    ax[0].set_title('Class Distribution Before Oversampling')
    ax[1].bar(['0', '1'], y_resampled.value_counts().values)
    ax[1].set_title('Class Distribution After Oversampling')
    plt.show()

oversampling()

#visualization of numerical features
#Replace the value Yes, Male as 1 and No, Female as 0
df = pd.read_csv('oversampled_data.csv')
df =  df[df.columns].replace({'No, borderline diabetes':'No','Yes (during pregnancy)':'Yes' })
df =  df[df.columns].replace({'Yes':1, 'No':0, 'Male':1,'Female':0})

def kFoldCrossValidation():
    # Convert categorical variables to numerical using one-hot encoding
    cat_cols = ['AgeCategory', 'Race', 'GenHealth']
    df_encoded = pd.get_dummies(df, columns=cat_cols)

    # Define X and y
    X = df_encoded.drop('HeartDisease', axis=1).values
    y = df_encoded['HeartDisease'].values

    # Define number of folds for K-fold cross-validation
    k_folds = 5

    # Create a KFold object
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Create an empty list to store the test scores
    test_scores = []

    #List the model name and
    model_name = ["K Nearest Neighbors model", "Decision Tree model", "Ensemble model"]
    models = [KNeighborsClassifier(), DecisionTreeClassifier(), VotingClassifier(estimators=[('knn', KNeighborsClassifier()), ('dt', DecisionTreeClassifier())])]
    print("\n\nThe test score, mean test score and standard deviation test score for K Fold Cross Validation for 3 model")
    # Iterate over the K folds
    for i,model in enumerate(models):
        test_scores = []
        for train_index, test_index in kf.split(X):
            # Split the data into train and test sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Evaluate the model on the test data and append the score to the list
            test_score = model.score(X_test, y_test)
            test_scores.append(test_score)

        # Print the mean and standard deviation of the test scores
        #the term "test scores" refers to the performance metric (i.e., R^2 score) obtained by fitting and
        # evaluating a linear regression model on the test data for each fold of a
        # 10-fold cross-validation procedure.
        print('\n')
        
        print(model_name[i], ": test scores:", test_scores)
        print(model_name[i], ": mean test score:", np.mean(test_scores))
        print(model_name[i], ": standard deviation of test scores:", np.std(test_scores))

kFoldCrossValidation()


def trainNtest():                                      
    #Preparing data for training and test
    #Define feature vector and target variable (class)
    #Select Features
    features = df.drop(columns =['HeartDisease'], axis = 1)
    #Select Target 
    target = df['HeartDisease']

    #Split data into separate training and test set
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle = True, test_size = 0.2, random_state = 44)

    print('\n\n')
    print('Shape of training feature:', X_train.shape)
    print('Shape of testing feature:', X_test.shape)
    print('Shape of training label:', y_train.shape)
    print('Shape of testing label:', y_test.shape)


    #data preprocessing
    #   encoding
    transformer = make_column_transformer((OneHotEncoder(sparse_output=False), ['AgeCategory', 'Race', 'GenHealth']),remainder='passthrough')

    # Encode training data 
    transformed_train = transformer.fit_transform(X_train)
    transformed_train_data = pd.DataFrame(transformed_train, columns=transformer.get_feature_names_out())

    # Concat the two tables
    transformed_train_data.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_train = pd.concat([transformed_train_data, X_train], axis=1)

    # Remove old columns
    X_train.drop(['AgeCategory', 'Race', 'GenHealth'], axis = 1, inplace = True)

    # Encode test data 
    transformed_test = transformer.fit_transform(X_test)
    transformed_test_data = pd.DataFrame(transformed_test, columns=transformer.get_feature_names_out())

    # Concat the two tables
    transformed_test_data.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    X_test = pd.concat([transformed_test_data, X_test], axis=1)

    # Remove old columns
    X_test.drop(['AgeCategory', 'Race', 'GenHealth'], axis = 1, inplace = True)

    #standardization
    scaler = StandardScaler()

    # Scale training data
    X_train = scaler.fit_transform(X_train)

    # Scale test data
    X_test = scaler.fit_transform(X_test)


    #modelling
    def evaluate_model(model, x_test, y_test):
        # Predict Test Data 
        y_pred = model.predict(x_test)

        # Calculate accuracy, precision, recall, f1-score, and kappa score
        acc = metrics.accuracy_score(y_test, y_pred)
        prec = metrics.precision_score(y_test, y_pred)
        rec = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        kappa = metrics.cohen_kappa_score(y_test, y_pred)

        # Calculate area under curve (AUC)
        y_pred_proba = model.predict_proba(x_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        # Display confussion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)

        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}

    
    #building model for KNN, Decision Tree and Ensemble model
    print("\n\nThe accuracy for train set and test set for KNN model, Decision Tree model and Ensemble model")
    # Create and fit a KNN model on the oversampled data using KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train, y_train)
    # Create and fit a Decision Tree model on the oversampled data using DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    # Create and fit a Ensemble model on the oversampled data using VotingClassifier
    ensemble = VotingClassifier(estimators=[('knn', knn), ('dt', dt)], voting='soft')
    ensemble.fit(X_train, y_train)

    
    # Evaluate accuracy of KNN model on train and test set
    y_train_pred_knn = knn.predict(X_train)
    y_test_pred_knn = knn.predict(X_test)
    print("KNN model train set accuracy: ", accuracy_score(y_train, y_train_pred_knn))
    print("KNN model test set accuracy: ", accuracy_score(y_test, y_test_pred_knn))
    print('\n')

    # Evaluate accuracy of Decision Tree model on train and test set
    y_train_pred_dt = dt.predict(X_train)
    y_test_pred_dt = dt.predict(X_test)
    print("Decision tree model train set accuracy: ", accuracy_score(y_train, y_train_pred_dt))
    print("Decision tree model test set accuracy: ", accuracy_score(y_test, y_test_pred_dt))
    print('\n')

    # Evaluate accuracy of Ensemble model (KNN model and Decision Tree model) on train and test set  
    y_train_pred_ensemble = ensemble.predict(X_train)
    y_test_pred_ensemble = ensemble.predict(X_test)
    print('Ensemble model train set accuracy: ', accuracy_score(y_train, y_train_pred_ensemble))
    print('Ensemble model test set accuracy: ', accuracy_score(y_test, y_test_pred_ensemble))
    print('\n')


    # Evaluate Model
    knn_eval = evaluate_model(knn, X_test, y_test)
    #Print result
    print("\nThe evaluation the performance of KNeighborsClassifier model")
    print("Test set evaluation:")
    print('Accuracy:', knn_eval['acc'])
    print('Precision:', knn_eval['prec'])
    print('Recall:', knn_eval['rec'])
    print('F1 Score:', knn_eval['f1'])
    print('Cohens Kappa Score:', knn_eval['kappa'])
    print('Area Under Curve:', knn_eval['auc'])
    print('Confusion Matrix:\n', knn_eval['cm'])

    # Evaluate Model
    dt_eval = evaluate_model(dt, X_test, y_test)
    # Print result
    print("\nThe evaluation the performance of Decision Tree Classifier model")
    print("Test set evaluation:")
    print('Accuracy:', dt_eval['acc'])
    print('Precision:', dt_eval['prec'])
    print('Recall:', dt_eval['rec'])
    print('F1 Score:', dt_eval['f1'])
    print('Cohens Kappa Score:', dt_eval['kappa'])
    print('Area Under Curve:', dt_eval['auc'])
    print('Confusion Matrix:\n', dt_eval['cm'])

    # Evaluate Model
    ensemble_eval = evaluate_model(ensemble, X_test, y_test)
    # Print result
    print("\nThe evaluation the performance of Ensemble Model")
    print("Test set evaluation:")
    print('Accuracy:', ensemble_eval['acc'])
    print('Precision:', ensemble_eval['prec'])
    print('Recall:', ensemble_eval['rec'])
    print('F1 Score:', ensemble_eval['f1'])
    print('Cohens Kappa Score:', ensemble_eval['kappa'])
    print('Area Under Curve:', ensemble_eval['auc'])
    print('Confusion Matrix:\n', ensemble_eval['cm'])

  
    #visualization of comparision through evaluation metrics and ROC curve between KNeighborsClassifier and DecisionTreeClassifier
    # Intitialize figure with two plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    fig.set_figheight(7)
    fig.set_figwidth(14)
    fig.set_facecolor('white')

    # First plot
    ## set bar size
    barWidth = 0.2
    dt_score = [dt_eval['acc'], dt_eval['prec'], dt_eval['rec'], dt_eval['f1'], dt_eval['kappa']]
    knn_score = [knn_eval['acc'], knn_eval['prec'], knn_eval['rec'], knn_eval['f1'], knn_eval['kappa']]
    esem_score = [ensemble_eval['acc'], ensemble_eval['prec'], ensemble_eval['rec'], ensemble_eval['f1'], ensemble_eval['kappa']]
    ## Set position of bar on X axis
    r1 = np.arange(len(dt_score))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    ## Make the plot
    ax1.bar(r1, dt_score, width=barWidth, edgecolor='white', label='Decision Tree')
    ax1.bar(r2, knn_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbours')
    ax1.bar(r3, esem_score, width=barWidth, edgecolor='white', label='Ensemble')
    ## Configure x and y axis
    ax1.set_xlabel('Metrics', fontweight='bold')
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
    ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(dt_score))], )
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_ylim(0, 1)
    ## Create legend & title
    ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
    ax1.legend()


    # Second plot
    ## Comparing ROC Curve
    ax2.plot(dt_eval['fpr'], dt_eval['tpr'], label='Decision Tree, auc = {:0.5f}'.format(dt_eval['auc']))
    ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='K-Nearest Nieghbour, auc = {:0.5f}'.format(knn_eval['auc']))
    ax2.plot(ensemble_eval['fpr'], ensemble_eval['tpr'], label='Ensemble, auc = {:0.5f}'.format(ensemble_eval['auc']))
    ## Configure x and y axis
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')
    ## Create legend & title
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc=4)
    plt.show()



trainNtest()


