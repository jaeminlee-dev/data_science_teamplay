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

# target은 성별을 0, 1로 변환한 값으로 기존과 동일해서 한개만 받음


def run_with_lda(lda_data1, lda_data2, target, types1, types2):
    X = np.concatenate([lda_data1, lda_data2], axis=1)
    y = target
    가우시안 = GaussianNB()
    가우시안.fit(X, y)
    show(가우시안, X, y, x_label=f'{types1} LDA Component 1', y_label=f'{types2}LDA Component 2', title='Naïve Bayes LDA')


def show(가우시안, X, y, x_label='키 (cm)', y_label='몸무게 (kg)', title='Naïve Bayes'):
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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
    pass
