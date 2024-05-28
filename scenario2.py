# 2. 남자/여자 허리둘레와 머리둘레를 PCA 한 차원축소 값을 구한다.
#     1. PCA한 허리둘레와 머리둘레를 데이터를 Logistic regression 한다.
from matplotlib import pyplot as plt
import pandas as pd


def run():
    pass

if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)