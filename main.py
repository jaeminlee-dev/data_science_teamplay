from sklearn.decomposition import PCA
import logistic as logistic
import naive as naive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pca import pca
sns.set()


if __name__ == '__main__':
    # Matplotlib에서 나눔 폰트 사용 설정
    plt.rc('font', family='AppleGothic')

    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')

    df_man = df[df['성별'] == '남']
    df_woman = df[df['성별'] == '여']
    df_man = df_man.drop(['성별', '측정일자'], axis=1)
    df_woman = df_woman.drop(['성별', '측정일자'], axis=1)

    attribute = ['키', '몸무게', '머리둘레']
    nd_man = df_man[attribute].values.reshape(-1, attribute.__len__())
    nd_woman = df_woman[attribute].values.reshape(-1, attribute.__len__())

    pca_man = pca(nd_man)
    pca_woman = pca(nd_woman)

    # sampleData = np.array([[175, 70], [160, 55]])
    # pca.PCA.transform(sampleData)

    naive.run(df)
    logistic.run(df)
