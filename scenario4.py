# 4. 남자/여자 키와 몸무게 LDA 값을 구한다. 허리둘레와 머리둘레 LDA 값을 구한다.
#     1. 두 데이터를  Naïve Bayes Classification 한다.
from matplotlib import pyplot as plt
import logistic as logistic
import naive as naive
import lda as lda
import pandas as pd
from itertools import combinations, chain

accuracy_weight_naive={}
accuracy_weight_logistic={}

def run(df):
    # 모든 경우의 수
    types = ['키', '몸무게', '머리둘레', 
             '목둘레', '희망치수신발',  '허리둘레(윗허리)',
             '어깨가쪽사이길이'
             ]

    #types = ['키', '몸무게', '희망치수신발']

    joinKeys = [list(arr) for arr in list(chain.from_iterable(combinations(types, size) for size in range(1, len(types) + 1)))]

    # 결과 출력
    print("학습 예정 조합 목록입니다.")
    for idx, keys in enumerate(joinKeys):
        print(f'[#{idx}] {keys}')

    result_naive = []
    result_logistic = []

    print("학습을 시작합니다")
    for idx, keys in enumerate(joinKeys):
        print(f'(학습 시작) [#{idx}] {keys}')
        x_train, x_test, y_train, y_test = lda.run(df, keys)

        ## Bayes
        accuracy = naive.accuracy_train_test(x_train, x_test, y_train, y_test)

        if(len(keys)==1):
            accuracy_weight_naive[keys[0]] = accuracy
        
        result_naive.append({'idx':idx, 'accuracy': accuracy, 'keys':sorted(keys, key=lambda x: accuracy_weight_naive[x] if x in accuracy_weight_naive else idx, reverse=True)})

        ## logistic
        accuracy = logistic.accuracy_train_test(x_train, x_test, y_train, y_test)

        if(len(keys)==1):
            accuracy_weight_logistic[keys[0]] = accuracy
        
        result_logistic.append({'idx':idx, 'accuracy': accuracy, 'keys':sorted(keys, key=lambda x: accuracy_weight_logistic[x] if x in accuracy_weight_logistic else idx, reverse=True)})

    print(f"# Bayes 총 결과 개수 : {len(result_naive)} #")

    result_naive.sort(key=lambda x: x['accuracy'], reverse=False)

    print(" Bayes 하위 결과 10위")
    for rank, token in enumerate(result_naive):
        if rank>=10:
            break;
        print(f'#{rank+1}위 => {token}')

    result_naive.sort(key=lambda x: x['accuracy'], reverse=True)

    print(" Bayes 상위 결과 10위")
    for rank, token in enumerate(result_naive):
        if rank>=10:
            break;
        print(f'#{rank+1}위 => {token}')

    print(f"# Logistic 총 결과 개수 : {len(result_logistic)} #")

    result_logistic.sort(key=lambda x: x['accuracy'], reverse=False)

    print(" Logistic 하위 결과 10위")
    for rank, token in enumerate(result_logistic):
        if rank>=10:
            break;
        print(f'#{rank+1}위 => {token}')

    result_logistic.sort(key=lambda x: x['accuracy'], reverse=True)

    print(" Logistic 상위 결과 10위")
    for rank, token in enumerate(result_logistic):
        if rank>=10:
            break;
        print(f'#{rank+1}위 => {token}')

    print("Bayes 각 요소별 순위")
    sorted_items = sorted(accuracy_weight_naive.items(), key=lambda item: item[1], reverse=True)
    print({key: value for key, value in sorted_items});

    # print("Logistic 각 요소별 순위")
    # sorted_items = sorted(accuracy_weight_logistic.items(), key=lambda item: item[1], reverse=True)
    # print({key: value for key, value in sorted_items});

if __name__ == '__main__':
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus']=False
    df = pd.read_csv(
        'content\공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
