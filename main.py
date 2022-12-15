from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


def createDataFrame() -> pd.DataFrame:
    '''This function takes data from 2 csv files and create dataframe with 2 columns'''
    df1 = pd.read_csv(r"D:\rtfiles\rose.csv", sep=',',
                      header=None, encoding='UTF-16')
    df2 = pd.read_csv(r"D:\rtfiles\tulip.csv", sep=',',
                      header=None, encoding='UTF-16')
    df = pd.concat([df1, df2], ignore_index=True)
    df.drop(1, axis=1, inplace=True)
    df.rename(columns={0: 'AbsolutePath', 2: 'DatasetClass'}, inplace=True)
    return df


def add_mark(df: pd.DataFrame) -> None:
    '''This function adds third column in dataframe - mark of image(1 or 0)'''
    value = []
    for item in df['DatasetClass']:
        if item == 'rose':
            value.append(0)
        else:
            value.append(1)
    df['mark'] = value


def add_hwcColumns(df: pd.DataFrame) -> None:
    '''This function adds to dataframe 3 columns: height, width and channels of the image'''
    img_width = []
    img_height = []
    img_channel = []
    for item in df['AbsolutePath']:
        img = cv2.imread(item)
        img_height.append(img.shape[0])
        img_width.append(img.shape[1])
        img_channel.append(img.shape[2])
    df['height'] = img_height
    df['width'] = img_width
    df['channel'] = img_channel


def mark_filter(df: pd.DataFrame, class_mark: int) -> pd.DataFrame:
    '''This function selects all images with mark and returns them'''
    return df[df['mark'] == class_mark]


def whm_filter(df: pd.DataFrame, class_mark: int, max_width: int, max_height: int) -> pd.DataFrame:
    '''This function selects all images with width and height less given and the same mark and returns them'''
    return df[(df.mark == class_mark) & (df.height <= max_height) & (df.width <= max_width)]


def group_mp(df: pd.DataFrame, class_mark: int) -> None:
    '''This function groops dataframe by new column (number of pixels) and shows information about that'''
    df = mark_filter(df, class_mark)
    img_pixels = []
    for item in df['AbsolutePath']:
        img = cv2.imread(item)
        img_pixels.append(img.size)
    df['pixels'] = img_pixels
    df.groupby('pixels').count()
    print(df.pixels.describe())


def create_histogram(df: pd.DataFrame, class_mark: int) -> List[np.ndarray]:
    '''This function creates histogram and returns it'''
    df = mark_filter(df, class_mark)
    df = df.sample()
    for item in df['AbsolutePath']:
        path = item
    img = cv2.imread(path)
    array = []
    for number in range(0, 3):  # blue green reds
        hist = cv2.calcHist([img], [number], None, [256], [0, 256])
        array.append(hist)
    return array


def histogram_rendering(df: pd.DataFrame, class_mark: int) -> None:
    '''This function draws histogram'''
    hist = create_histogram(df, class_mark)
    plt.plot(hist[0], color='b')
    plt.plot(hist[1], color='g')
    plt.plot(hist[2], color='r')
    plt.title('Image Histogram For Blue, Green, Red Channels')
    plt.xlabel("Intensity")
    plt.ylabel("Number of pixels")
    plt.show()


if __name__ == '__main__':
    df = createDataFrame()
    add_mark(df)
    add_hwcColumns(df)
    group_mp(df, 1)
    histogram_rendering(df, 1)
