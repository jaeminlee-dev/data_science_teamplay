import logistic as logistic
import naive as naive
import pca as pca
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    sns.set()
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    
    df_man = df[df['성별'] == '남']
    
