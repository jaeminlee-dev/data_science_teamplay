# 1. 남자/여자 키와 몸무게를 PCA 한 차원축소 값을 구한다.
#     1. PCA한 키와 몸무게 데이터를 Logistic regression 한다.

import logistic as logistic
import naive as naive
import pca as pca

def run(df):
    pca.run(df)
    pass