from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


def run(df, types=['키', '몸무게']):
    scaled = StandardScaler().fit_transform(df[types])
    target = df['성별'].replace({"남": 0, "여": 1}).values

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(scaled, target)
    lda_data = lda.transform(scaled)

    show(lda_data, target)
    return lda_data, target


def show(lda_data, target):
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


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
    pass
