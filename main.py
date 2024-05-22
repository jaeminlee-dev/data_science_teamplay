import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import naive as naive
import logistic as logistic
import pca as pca

# Matplotlib에서 나눔 폰트 사용 설정
plt.rc('font', family='NanumGothic')

df = pd.read_csv('2학년1학기/datascience/content/공군_신체정보_남녀혼합.csv', encoding='cp949')

pca.run(df)
naive.run(df)
logistic.run(df)
