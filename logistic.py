import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
def run(df):
    # 성별에 따라 데이터 분리
    남자 = df[df['성별'] == '남']
    여자 = df[df['성별'] == '여']

    키_남자 = 남자['키']
    키_여자 = 여자['키']

    # 레이블 생성
    레이블_남자 = np.zeros_like(키_남자)
    레이블_여자 = np.ones_like(키_여자)

    # 매개변수 정의
    바이어스 = -25  # 시그모이드 곡선을 이동시키는 바이어스
    가중치 = 0.15  # 시그모이드 곡선을 스케일링하는 가중치

    def 시그모이드(입력값):
        # z는 입력값의 선형 함수
        z = 가중치 * 입력값 + 바이어스
        return 1 / (1 + np.exp(-z))

    # 키 범위 내에서 일정 간격으로 값 생성
    입력_값들 = np.linspace(150, 190, 300)

    # 입력 값들에 대한 시그모이드 값 계산
    시그모이드_값들 = 시그모이드(입력_값들)

    # 플로팅
    plt.plot(키_남자, 레이블_남자, 'o', label='남자')
    plt.plot(키_여자, 레이블_여자, 'o', label='여자')
    plt.plot(입력_값들, 시그모이드_값들, color='red', label='시그모이드 함수')

    plt.xlabel('키')
    plt.ylabel(r'$\sigma(z)$')
    plt.legend()
    plt.show()
