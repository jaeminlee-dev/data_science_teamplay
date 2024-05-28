# 1. 남자/여자 키와 몸무게를 PCA 한 차원축소 값을 구한다.
#     1. PCA한 키와 몸무게 데이터를 Logistic regression 한다.

from matplotlib import pyplot as plt
import logistic as logistic
import naive as naive
import pca as pca
import pandas as pd


def run(df):
    typesList = []
    typesList.append(['키', '몸무게', '허리둘레(윗허리)', '어깨가쪽사이길이', '머리둘레', '목둘레', '희망치수신발'])
    typesList.append(['키', '몸무게', '허리둘레(윗허리)', '머리둘레', '목둘레', '희망치수신발'])
    typesList.append(['키', '몸무게', '허리둘레(윗허리)', '희망치수신발'])
    typesList.append(['허리둘레(윗허리)', '몸무게', '어깨가쪽사이길이', '희망치수신발'])
    typesList.append(['허리둘레(윗허리)', '키', '어깨가쪽사이길이', '희망치수신발'])
    typesList.append(['키', '허리둘레(윗허리)', '희망치수신발'])
    typesList.append(['몸무게', '허리둘레(윗허리)', '희망치수신발'])
    typesList.append(['키', '몸무게', '머리둘레', '목둘레', '희망치수신발'])
    typesList.append(['머리둘레', '목둘레'])
    typesList.append(['머리둘레', '허리둘레(윗허리)'])
    typesList.append(['희망치수신발'])
    typesList.append(['키', '몸무게', '희망치수신발'])
    typesList.append(['키', '몸무게'])
    typesList.append(['키'])
    #types = ['키', '몸무게']
    #types = ['키']
    
    for types in typesList:
        pcaed_data = pca.run(df, types)
        남자 = df[df['성별'] == '남']
        여자 = df[df['성별'] == '여']
        logistic.run_with_pca(pcaed_data, 남자.shape[0], 여자.shape[0], types)
    # logistic.run_with_pca(man_pcaed_data, woman_pcaed_data, types)


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
