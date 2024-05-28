# 4. 남자/여자 키와 몸무게 LDA 값을 구한다. 허리둘레와 머리둘레 LDA 값을 구한다.
#     1. 두 데이터를  Naïve Bayes Classification 한다.
from matplotlib import pyplot as plt
import pandas as pd


def run():
    pass


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
