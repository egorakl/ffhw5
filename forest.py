import tree
import numpy as np
import matplotlib.pyplot as plt


# def bootstrap(x, y, quan):
#     r = np.random.random_integers(x.shape[0], size=(x.shape[0]//quan))
#     return x[r, :], y[r]

def learn_forest_learn(x, y, quan, depth):  # quan и depth -- кол-во и глубина деревьев
    forest = []
    for i in range(quan):
        r = np.random.random_integers(x.shape[0]-1, size=(x.shape[0]//quan))
        xx = x[r, :]
        yy = y[r].T
        # xx, yy = bootstrap(x, y, quan)
        # print(xx)
        # print(yy)
        forest.append(tree.build_tree(xx, yy, max_depth=depth))
    return forest


def predict_forest_predict(forest, x):
    k = len(forest)
    s = 0.0
    for i in range(k):
        s += tree.predict(forest[i], x) / k
    return s

# Для y=x^2
#
# n = 1000
#
# x = np.random.normal(0, 1, size=(n, 2))
# y_true = x[:, 0]**2
# y = y_true + np.random.normal(0, 0.5, n)
#
# plt.plot(y_true, y, 'o')
#
# x_test = np.random.normal(0, 1, size=(n, 2))
# y_test = x_test[:, 0]**2
#
# forest = learn_forest_learn(x, y, 10, 7)
# y_pred = predict_forest_predict(forest, x_test)
#
# plt.plot(y_test, y_pred, 'x')
# print(np.std(y_test - y_pred))
# plt.plot(plt.xlim(), plt.xlim(), 'k', lw=0.5)
#
# plt.show()
