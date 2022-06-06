"""
Обучающаяся нейронная сеть. Нейроны расположены в порядке: 2-3-2-1
Их можно добавлять/убавлять (через изменение весов)
Обучение идет через метод обратного распространения ошибки
"""
import numpy as np
# входные данные
INP = np.array([[0.6],
                [0.7]])
# истинное значение
VAL_CORR = 0.9
# кол-во проходов
EPOCHS = 10
# скорость обучения
ALPHA = 0.3
# момент - для коррекции весовых коэффициентов
GAMMA = 0.1
# начальные значения весов между нейронами (нейроны в таком порядке: 2-3-2-1)
W0 = np.array([[-1, 1],
               [2.5, 0.4],
               [1, -1.5]])
W1 = np.array([[2.2, -1.4, 0.56],
               [0.34, 1.05, 3.1]])
W2 = np.array([[0.75, -0.22]])
WEIGHTS = [W0, W1, W2]

# функция активации (сигмоида)
def act(x):
   return 1 / (1 + np.exp(-x))

# производная функции активации
def act_p(x):
    return act(x) * (1 - act(x))

# вычисляет значения на входе и на выходе каждого нейрона
def calc_in_out(weights):
    vector_act = np.vectorize(act)
    inputs = []
    outputs = [INP]
    for weight in weights:
        inp = np.dot(weight, outputs[-1])
        # запомним значения на входе, они буду нужны для
        # расчета обратного распространения ошибки
        inputs.append(inp)
        # применяем функцию активации, чтоб получить выходное значение
        outputs.append(vector_act(inp))
    return inputs, outputs

# вычисляет ошибки для каждого нейрона
# при обратном распространении ошибки ошибка для каждого элемента вычисляется
# взвешенным суммированием ошибок, приходящих к нему нейронов от следующего
# слоя и умножения на производную функции активации
def calc_err(err_exit, inputs, weights):
    vector_act_p = np.vectorize(act_p)
    errors = [err_exit]
    # обратный проход, поэтому движемся с конца
    for i in range(len(weights) - 1, 0, -1):
        a = np.dot(errors[-1].T, weights[i])
        r = vector_act_p(inputs[i - 1])
        # умножаем на производную функции активации
        errors.append(a.T * r)
    return errors

# вычисляет величину поправки для каждого веса
def calc_corr(errors, outputs, corrects_prev):
    outputs = outputs[::-1]
    corrects = []
    for i in range(len(errors)):
        try:
            adj = GAMMA * corrects_prev[i]
        except:
            adj = 0
        res = ALPHA * np.dot(errors[i], outputs[i + 1].T) + adj
        corrects.append(res)
    return corrects

# вычисляет новые веса
def calc_weight(corrects, weights):
    corrects = corrects[::-1]
    return [weights[i] + corrects[i] for i in range(len(corrects))]

# функция обучения нейронной сети
def learn(weights_prev, corrects_prev):
    # ПРЯМОЙ ПРОХОД
    # вычисляем входные и выходные значения нейронов
    inputs, outputs = calc_in_out(weights_prev)
    # вычисляем выходную ошибку
    err_exit = (VAL_CORR - outputs[-1][0, 0]) * act_p(inputs[-1])

    # ОБРАТНЫЙ ПРОХОД
    # вычисляем ошибки для каждого нейрона
    errors = calc_err(err_exit, inputs, weights_prev)
    # вычисляем поправку для каждого веса
    corrects = calc_corr(errors, outputs, corrects_prev)
    # вычисляем новые веса
    weights = calc_weight(corrects, weights_prev)
    # возвращаем: итоговое значение прохода, поправки, веса
    return outputs[-1][0, 0], corrects, weights

weights_prev = WEIGHTS
corrects_prev = []
for epoch in range(EPOCHS):
    res = learn(weights_prev, corrects_prev)
    print(f'Epoch {epoch + 1} from {EPOCHS}: {res[0]}, target: {VAL_CORR}')
    corrects_prev = res[1]
    weights_prev = res[2]
