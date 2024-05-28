from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def run(df, types=['키', '몸무게']):
    X = df[types].values
    y = df['성별'].apply(lambda x: 0 if x == '남' else 1).values
    가우시안 = GaussianNB()
    가우시안.fit(X, y)
    show(가우시안, X, y)


def show(가우시안, X, y):
    # 시각화
    rng = np.random.RandomState(0)
    Xnew = np.array([[rng.uniform(X[:, 0].min(), X[:, 0].max()),
                    rng.uniform(X[:, 1].min(), X[:, 1].max())] for _ in range(2000)])
    ynew = 가우시안.predict(Xnew)

    # 기존 데이터 표출
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu',
                edgecolor='k', label='기존 데이터')

    # 새 데이터 표출
    plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20,
                cmap='RdBu', alpha=0.1, label='생성된 데이터')

    # 라벨 세팅
    plt.xlabel('키 (cm)')
    plt.ylabel('몸무게 (kg)')
    plt.title('가우시안 나이브 베이지스 분류')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
    pass
