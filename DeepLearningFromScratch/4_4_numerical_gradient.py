#편미분 함수의 기울기를 구하는 코드

import numpy as np

def function(x):
    return x[0]**2 + x[1]**2
    #편미분 할 함수

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    #np.zeros_like(x) - x와 형상이 같고 그 원소가 모두 0인 배열을 만듭니다.

    for idx in range(x.size):
        tmp_val = x[idx]
        #for문은 리스트의 첫번째 요소를 idx에 대입한 후 아래의 명령을 수행합니다. 수행을 마쳤다면 이를 리스트의 마지막 요소까지 반복합니다.
        # x.size - x안에 있는 요소의 개수를 출력합니다.
        # range(x.size) - 0부터 x.size 미만의 숫자를 포함하는 리스트를 생성합니다.
        # 따라서 0 <= idx < x.size

        x[idx] = tmp_val + h
        fxh1 = f(x)
        #f(x+h) 계산
        #x의 해당 인덱스에 있는 요소가 tmp_val + h로 대체됩니다.
        #따라서 f(x)의 x에 tmp_val + h값이 들어가며 이를 fxh1으로 정의합니다.

        x[idx] = tmp_val - h
        fxh2 = f(x)
        # f(x-h) 계산
        # x의 해당 인덱스에 있는 요소가 tmp_val - h로 대체됩니다.
        # 따라서 f(x)의 x에 tmp_val - h값이 들어가며 이를 fxh1으로 정의합니다.

        grad[idx] = (fxh1 - fxh2) / (2*h)
        #grad의 해당 인덱스에 있는 요소(전부 0)를 (fxh1 - fxh2) / (2*h)로 변경합니다.
        x[idx] = tmp_val
        #for문이 리스트의 다음 요소를 idx에 대입하도록 x[idx]값을 원래대로 되돌립니다.
    return grad
    #변경된 grad를 출력합니다.

#example
print(numerical_gradient(function, np.array([3.0, 4.0])))
"""
grad = array([0, 0])

0 <= idx < 2

idx = 0일때
tmp_val = x[0] = 3.0
x[0] = tmp_val + h = 3.0 + h
따라서 array([3.0, 4.0])였던 x가 array([3.0 + h, 4.0])이 됩니다.
f(x) = x[0]**2 + x[1]**2 이므로
fxh1 = (3.0 + h)**2 + (4.0)**2
마찬가지로 fxh2 = (3.0 - h)**2 + (4.0)**2
grad[0] = (fxh1 - fxh2) / (2*h)
        = [{(3.0 + h)**2 + (4.0)**2} - {(3.0 - h)**2 + (4.0)**2}] / 2h
        = 6.
따라서 grad = grad = array([6., 0])


idx = 1일때
tmp_val = x[1] = 4.0
x[1] = tmp_val + h = 4.0 + h
따라서 array([3.0, 4.0])였던 x가 array([3.0, 4.0 + h])이 됩니다.
f(x) = x[0]**2 + x[1]**2 이므로
fxh1 = (3.0)**2 + (4.0 + h)**2
마찬가지로 fxh2 = (3.0)**2 + (4.0 - h)**2
grad[1] = (fxh1 - fxh2) / (2*h)
        = [{(3.0)**2 + (4.0 + h)**2} - {(3.0)**2 + (4.0 - h)**2}] / 2h
        = 8.
따라서 grad = grad = array([6., 8.])

반복을 마쳤으므로 for문을 빠져나가고 결과값 grad를 출력합니다.
"""