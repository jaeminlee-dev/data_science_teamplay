import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


def run(df, type='몸무게'):
    남자 = df[df['성별'] == '남']
    여자 = df[df['성별'] == '여']

    # 남자 필터링된 데이터와 여자 필터링된 데이터를 합친다.
    x = np.concatenate((남자[type].values, 여자[type].values))
    X = x.reshape(-1, 1)
    # 남자는 0, 여자는 1로 레이블링
    y = np.concatenate((np.zeros(남자.shape[0]), np.ones(여자.shape[0])))
    clf = LogisticRegression(solver='lbfgs').fit(X, y)
    show(clf, title=f'{type} 로지스틱 회귀 분석')


def run_with_pca(pcaed_data, shape1, shape2, types):
    # 남자 필터링된 데이터와 여자 필터링된 데이터를 합친다.

    x = pcaed_data
    y = np.concatenate((np.zeros(shape1), np.ones(shape2)))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)   

    # 남자는 0, 여자는 1로 레이블링
    clf = LogisticRegression(solver='lbfgs').fit(x_train, y_train)
    clf.fit(x_train, y_train)

    # 예측 및 평가
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{types} 모델 정확도: {accuracy}')
    #print(classification_report(y_test, y_pred))

    # 예측 확률 시각화
    # probabilities = clf.predict_proba(x_test)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(range(len(y_test)), probabilities[:, 1], color='red', label='여자일 확률')
    # plt.scatter(range(len(y_test)), probabilities[:, 0], color='blue', label='남자일 확률')
    # plt.axhline(y=0.5, color='red', linestyle='--', label='결정 경계')
    # plt.title('로지스틱 회귀 분석 (PCA 주성분 점수 사용)')
    # plt.xlabel('샘플 인덱스')
    # plt.ylabel('여자일 확률')
    # plt.legend()
    # plt.show()
    #show(clf, title=f'{types} 로지스틱 회귀 분석 (PCA)')


def show(clf, title='로지스틱 회귀 분석'):
    # TODO: 데이터의 형식에 따라 맞춤형으로 x값 설정
    x = np.arange(-100, 100, 1)
    y_male = clf.predict_proba(x.reshape(-1, 1))[:, 0]
    y_female = clf.predict_proba(x.reshape(-1, 1))[:, 1]

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_male * 100, label='남자일 확률', color='blue')
    plt.plot(x, y_female * 100, label='여자일 확률', color='red')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
    pass
