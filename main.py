
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scenario1 as scenario1
import scenario2 as scenario2
import scenario3 as scenario3
import scenario4 as scenario4


if __name__ == '__main__':
    sns.set()
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')

    scenario1.run(df)
    # scenario2.run(df)
    # scenario3.run(df)
    # scenario4.run(df)
