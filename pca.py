from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def run(df, sex='남', types=['키','희망치수신발']):
    df_filtered_sex = df[df['성별'] == sex]
    df_reshaped = df_filtered_sex[types].values.reshape(-1, len(types))
    pca = PCA(n_components=1)
    pca_data = pca.fit_transform(df_reshaped)
    #show(pca_data)
    return pca_data

def run(df, types=['키','희망치수신발']):
    df_reshaped = df[types].values.reshape(-1, len(types))
    pca = PCA(n_components=1)
    pca_data = pca.fit_transform(df_reshaped)
    # show(pca_data)
    return pca_data

def show(pca_data):
    # 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_data, [0]*len(pca_data), alpha=0.5)
    plt.xlabel('PCA Component 1')
    plt.yticks([])  # Y축 값 숨기기
    plt.show()


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
    pass
