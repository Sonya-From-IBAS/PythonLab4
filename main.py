import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def createDataFrame():
    df1 = pd.read_csv("rose.csv", sep=',', header=None, encoding='UTF-16')
    df2 = pd.read_csv("tulip.csv", sep=',', header=None, encoding='UTF-16')
    df = pd.concat([df1, df2], ignore_index=True)
    df.drop(1, axis=1, inplace=True)
    df.rename(columns={0: 'AbsolutePath', 2: 'DatasetClass'}, inplace=True)
    return df


def add_mark(df: pd.DataFrame) -> None:
    value = []
    for item in df['DatasetClass']:
        if item == 'rose':
            value.append(0)
        else:
            value.append(1)
    df['mark'] = value


df = createDataFrame()
add_mark(df)
print(df)
