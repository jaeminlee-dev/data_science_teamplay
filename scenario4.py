# 4. 남자/여자 키와 몸무게 LDA 값을 구한다. 허리둘레와 머리둘레 LDA 값을 구한다.
#     1. 두 데이터를  Naïve Bayes Classification 한다.
from matplotlib import pyplot as plt
import logistic as logistic
import naive as naive
import lda as lda
import pandas as pd


def run(df):
    types1 = ['키', '몸무게']
    types2 = ['허리둘레(윗허리)', '머리둘레']

    lda_data1, target1 = lda.run(df, types1)
    lda_data2, target2 = lda.run(df, types2)

    naive.run_with_lda(lda_data1, lda_data2, target1, types1, types2)


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
