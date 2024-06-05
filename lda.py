from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def run(df, types=['키', '몸무게']):
    target = df['성별'].map({"남": 0, "여": 1})
    x_train, x_test, y_train, y_test = train_test_split(df[types], target, test_size=0.2, random_state=42)
    
    lda = LinearDiscriminantAnalysis(n_components=1)
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)

    ## 각각 lda를 적용한 데이터셋이다
    return x_train_lda, x_test_lda, y_train, y_test;


def show(lda_data, target):
    markers = ['^', 's']
    colors = ['blue', 'green']
    target_names = ['man', 'girl']

    for i in range(2):
        plt.scatter(lda_data[target == i][:, 0], [0]*len(lda_data[target == i][:, 0]),
                    marker=markers[i], color=colors[i], label=target_names[i])

    plt.legend(loc='upper right')
    plt.xlabel('lda_component_1')
    plt.ylabel('lda_component_2')
    # plt.show()


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
    pass
