import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from collections import namedtuple

Node = namedtuple('Node', ('feature', 'value', 'impurity', 'left', 'right'))  # Узел, кот. ссылается на 2 других узла
Leaf = namedtuple('Leaf', ('value', 'x', 'y'))  # Лист


def build_tree(x, y, depth=1, max_depth=np.inf):

    if depth >= max_depth or criteria(y) < 1e-6:
        return Leaf(np.mean(y), x, y)

    feature, value, impurity = find_best_split(x, y)
    (x_left, y_left), (x_right, y_right) = partition(x, y, feature, value)
    left = build_tree(x_left, y_left, depth+1, max_depth)
    right = build_tree(x_right, y_right, depth+1, max_depth)
    root = Node(feature, value, impurity, left, right)
    return root


def partition(x, y, feature, value):
    i_right = x[:, feature] >= value
    i_left = np.logical_not(i_right)
    return (x[i_left], y[i_left]), (x[i_right], y[i_right])


def criteria(y):
    return np.std(y)**2


def impurity(y_left, y_right):
    size = y_left.size + y_right.size
    h = (y_left.size * criteria(y_left) + y_right.size * criteria(y_right)) / size
    return h


def f(value, feature, x, y):
    (_, y_left), (_, y_right) = partition(x, y, feature, value)
    return impurity(y_left, y_right)


def find_best_split(x, y):
    best_feature, best_value, best_impurity = 0, x[0, 0], np.inf
    for feature in range(x.shape[1]):
        x_i_sorted = np.sort(x[:, feature])
        result = minimize_scalar(
            f,
            args=(feature, x, y),
            method='Bounded',
            bounds=(x_i_sorted[1], x_i_sorted[-1]),
        )
        value = result.x
        impurity = result.fun
        if impurity < best_impurity:
            best_feature, best_value, best_impurity = feature, value, impurity

    return best_feature, best_value, best_impurity


def predict(tree, x):
    y = np.empty(x.shape[0])
    for i, row in enumerate(x):
        node = tree
        while isinstance(node, Node):
            if row[node.feature] >= node.value:
                node = node.right
            else:
                node = node.left
        y[i] = node.value
    return y


# y = 2*x0 + 1
# n = 1000
# x = np.random.normal(0, 1, size=(n, 2))
# y = 2 * x[:, 0] + 1 + np.random.normal(0, 0.5, n)
# tree = build_tree(x, y)
# plt.subplot(121)
# plt.plot(y, predict(tree, x), 'o')
# x_test = np.random.normal(0, 1, size=(n,2))
# y_test = 2 * x_test[:, 0] + 1
# y_pred = predict(tree, x_test)
# plt.subplot(122)
# plt.plot(y_test, y_pred, 'v')
# plt.plot(plt.xlim(), plt.xlim(), 'k', lw=0.5)
# plt.show()





# n = 1000
#
# # Круг
# # x = np.random.normal(0, 1, size=(n, 2))
# # y = np.asarray(x[:, 0]**2 + x[:, 1]**2 <= 1, dtype=int)
# # tree = build_tree(x, y)
# # colors = np.array([[1., 0., 0.], [0., 0., 1.]])
# # plt.figure(figsize=(11, 5))
# # plt.subplot(121)
# # plt.scatter(*x.T, color=colors[y])
#
#
# # XOR
# x = np.random.normal(0, 1, size=(n, 2))
# y = np.asarray(x[:, 0] * x[:, 1] > 0, dtype=int)
#
# x_rand = np.random.normal(0, 1, size=(n//5, 2))
# y_rand = np.random.randint(2, size=n//5)
# x = np.concatenate((x, x_rand))
# y = np.concatenate((y, y_rand))
#
# tree = build_tree(x, y)
# colors = np.array([[1., 0., 0.], [0., 0., 1.]])
# plt.figure(figsize=(11, 5))
# plt.subplot(121)
# plt.scatter(*x.T, color=colors[y])
#
#
#
# x_test = np.random.normal(0, 1, size=(n, 2))
# y_pred = predict(tree, x_test).astype(np.int)
#
# plt.subplot(122)
# plt.scatter(*x_test.T, color=colors[y_pred], marker='v', s=20)
# plt.show()