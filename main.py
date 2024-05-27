import logistic as logistic
import naive as naive
import pca as pca
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    df_man = df[df['성별'] == '남']
    attribute = ['키', '몸무게', '머리둘레']
    # pca.PCA.transform(sampleData)

    # naive.run(df)
    # logistic.run(df)
    pca.run(df_man)
