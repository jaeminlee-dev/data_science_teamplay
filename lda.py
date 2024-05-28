from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


def run(df, types=['키', '몸무게']):
    scaled = StandardScaler().fit_transform(df[types])
    lda = LinearDiscriminantAnalysis(n_components=1)

    target = df['성별'].replace({"남": 0, "여": 1}).values
    lda.fit(scaled, target)
    lda_data = lda.transform(scaled)

    markers = ['^', 's']
    colors = ['blue', 'green']
    target_names = ['man', 'girl']

    for i in range(2):
        plt.scatter(lda_data[target == i][:, 0], [0]*len(lda_data[target == i][:, 0]),
                    marker=markers[i], color=colors[i], label=target_names[i])

    plt.legend(loc='upper right')
    plt.xlabel('lda_component_1')
    # plt.ylabel('lda_component_2')
    plt.show()


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
