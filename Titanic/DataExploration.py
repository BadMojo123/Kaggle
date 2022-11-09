import pandas as pd
import matplotlib.pyplot as plt
import scipy


def DataExploration(DataSet: pd.DataFrame):
    DataSet = pd.DataFrame(DataSet)
    # print('Data columns')
    # print(DataSet.columns)
    print('Data shape')
    print(DataSet.shape)

    pd.set_option("display.max.columns", None)
    print(DataSet.head())

    print('Nan check:')
    print(DataSet.isnull().values.any())
    print(DataSet.isnull().sum(axis=0))


    DataSet.plot(kind='kde')
    plt.show()
    print('Data description')
    print(DataSet.describe())
    print('Data Info')
    print(DataSet.info())

#
# full_data = pd.read_csv('train.csv')
# DataExploration(full_data)