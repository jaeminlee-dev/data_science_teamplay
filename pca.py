from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def run(df, sex='남'):
    df_man = df[df['성별'] == sex]

    df_man = df_man.drop(['성별', '측정일자'], axis=1)

    남자키몸무게 = df_man[['키', '몸무게']].values.reshape(-1, 2)

    pca = PCA(n_components=1)
    남자키몸무게PCA = pca.fit_transform(남자키몸무게)

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(남자키몸무게PCA, [0]*len(남자키몸무게PCA), alpha=0.5)
    plt.title('남자키몸무게PCA')
    plt.xlabel('PCA Component 1')
    plt.yticks([])  # Y축 값 숨기기
    plt.show()

if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
    pass