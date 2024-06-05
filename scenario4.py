# 4. 남자/여자 키와 몸무게 LDA 값을 구한다. 허리둘레와 머리둘레 LDA 값을 구한다.
#     1. 두 데이터를  Naïve Bayes Classification 한다.
from matplotlib import pyplot as plt
import logistic as logistic
import naive as naive
import lda as lda
import pandas as pd
from itertools import combinations, chain

accuracy_weight={}

def run(df):
    # 모든 경우의 수
    types = ['키', '몸무게', '머리둘레', 
             '목둘레', '화장', '젖가슴둘레',
             '배꼽수준허리둘레','엉덩이둘레', '샅높이',
             '희망치수신발', '윗가슴둘레(겨드랑이)', '허리둘레(윗허리)',
             '어깨가쪽사이길이', '팔길이', '등길이',
             '다리가쪽길이', '총장'
             ]

    joinKeys = [list(arr) for arr in list(chain.from_iterable(combinations(types, size) for size in range(1, len(types) + 1)))]

    # 결과 출력
    print("학습 예정 조합 목록입니다.")
    for idx, keys in enumerate(joinKeys):
        print(f'[#{idx}] {keys}')

    result = []

    print("학습을 시작합니다")
    for idx, keys in enumerate(joinKeys):
        print(f'(학습 시작) [#{idx}] {keys}')
        x_train, x_test, y_train, y_test = lda.run(df, keys)
        accuracy = naive.accuracy_train_test(x_train, x_test, y_train, y_test)

        if(len(keys)==1):
            accuracy_weight[keys[0]] = accuracy
        
        result.append({'idx':idx, 'accuracy': accuracy, 'keys':sorted(keys, key=lambda x: accuracy_weight[x], reverse=True)})

        print(f'(학습 종료) [#{idx}] {keys} => 정확도: {accuracy}')

    print(f"# 총 결과 개수 : {len(result)} #")

    result.sort(key=lambda x: x['accuracy'], reverse=False)

    print("하위 결과 10위")
    for rank, token in enumerate(result):
        if rank>=10:
            break;
        print(f'#{rank+1}위 => {token}')

    result.sort(key=lambda x: x['accuracy'], reverse=True)

    print("상위 결과 10위")
    for rank, token in enumerate(result):
        if rank>=10:
            break;
        print(f'#{rank+1}위 => {token}')
    # lda_data1, target1 = lda.run(df, types1)
    # lda_data2, target2 = lda.run(df, types2)

    # naive.run_with_lda(lda_data1, lda_data2, target1, types1, types2)


if __name__ == '__main__':
    plt.rc('font', family='AppleGothic')
    df = pd.read_csv(
        './content/공군_신체정보_남녀혼합.csv', encoding='cp949')
    run(df)
