# 1. 남자/여자 키와 몸무게를 PCA 한 차원축소 값을 구한다.
#     1. PCA한 키와 몸무게 데이터를 Logistic regression 한다.

from matplotlib import pyplot as plt
import logistic as logistic
import naive as naive
import pca as pca
import pandas as pd


def run(df):
    types = ['키', '몸무게']
    man_pcaed_data = pca.run(df, '남', types)
    woman_pcaed_data = pca.run(df, '여', types)
    logistic.run_with_pca(man_pcaed_data, woman_pcaed_data, types)


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
