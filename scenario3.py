# 남자/여자 키와 몸무게로 Naïve Bayes Classification 한다.
from matplotlib import pyplot as plt
import pandas as pd
import naive as naive


def run(df):
    naive.run(df)


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
