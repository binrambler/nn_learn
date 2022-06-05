import numpy as np

out0 = np.array([[0.6],
                [0.7]])
val_corr = 0.9
alpha = 0.3
w0 = np.array([[-1, 1],
               [2.5, 0.4],
               [1, -1.5]])
w1 = np.array([[2.2, -1.4, 0.56],
               [0.34, 1.05, 3.1]])
w2 = np.array([[0.75, -0.22]])
layers = [w0, w1, w2]

def act(x):
   return 1 / (1 + np.exp(-x))

def act_p(x):
    return act(x) * (1 - act(x))

def calc_in_out(out0, layers):
    vector_act = np.vectorize(act)
    inputs = []
    outputs = [out0]
    for layer in layers:
        inp = np.dot(layer, outputs[-1])
        outputs.append(vector_act(inp))
        inputs.append(inp)
    return inputs, outputs

def calc_err(err_exit, inputs, layers):
    vector_act_p = np.vectorize(act_p)
    errors = [err_exit]
    for i in range(len(layers) - 1, 0, -1):
        a = np.dot(errors[-1].T, layers[i])
        r = vector_act_p(inputs[i - 1])
        errors.append(a.T * r)
    return errors

def calc_corr(alpha, errors, outputs):
    outputs = outputs[::-1]
    corrects = []
    for i in range(len(errors)):
        corrects.append(alpha * np.dot(errors[i], outputs[i + 1].T))
    return corrects

inputs, outputs = calc_in_out(out0, layers)

err_exit = (val_corr - outputs[-1][0, 0]) * act_p(inputs[-1])
errors = calc_err(err_exit, inputs, layers)

# расчет корректировочных величин для весов
corrects = calc_corr(alpha, errors, outputs)
print(corrects)