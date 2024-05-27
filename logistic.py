import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def run(df, type='키'):
    남자 = df[df['성별'] == '남']
    여자 = df[df['성별'] == '여']
    x = np.concatenate((남자[type].values, 여자[type].values))
    X = x.reshape(-1, 1)
    y = np.concatenate((np.zeros(남자.shape[0]), np.ones(여자.shape[0])))

    clf = LogisticRegression(solver='lbfgs').fit(X, y)
    show(clf)


def show(clf):
    # TODO: 데이터의 형식에 따라 맞춤형으로 x값 설정
    x = np.arange(140, 200, 1)
    y_male = clf.predict_proba(x.reshape(-1, 1))[:, 0]
    y_female = clf.predict_proba(x.reshape(-1, 1))[:, 1]

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_male * 100, label='남자일 확률', color='blue')
    plt.plot(x, y_female * 100, label='여자일 확률', color='red')
    plt.title('로지스틱 회귀 분석')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
    pass
