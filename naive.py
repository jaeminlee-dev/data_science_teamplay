import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.naive_bayes import GaussianNB

def run(df):
    # Matplotlib에서 나눔 폰트 사용 설정
    plt.rc('font', family='AppleGothic')

    # 데이터 준비
    X = df[['키', '몸무게']].values
    y = df['성별'].apply(lambda x: 0 if x == '남' else 1).values

    # 가우시안 모델로 핏
    가우시안 = GaussianNB()
    가우시안.fit(X, y)

    # 새로운 데이터들 잔뜩 만들고 가우시안에 적용
    rng = np.random.RandomState(0)
    Xnew = np.array([[rng.uniform(X[:, 0].min(), X[:, 0].max()),
                    rng.uniform(X[:, 1].min(), X[:, 1].max())] for _ in range(2000)])
    ynew = 가우시안.predict(Xnew)

    # 기존 데이터 표출
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu', edgecolor='k', label='기존 데이터')

    # 새 데이터 표출
    plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1, label='생성된 데이터')

    # 라벨 세팅
    plt.xlabel('키 (cm)')
    plt.ylabel('몸무게 (kg)')
    plt.title('가우시안 나이브 베이지스 분류')
    plt.legend()
    plt.show()