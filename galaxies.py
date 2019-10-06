import forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

depth = 10
quan = 35

file = pd.read_csv("sdss_redshift.csv")
data_learn = file[["u", "g", "r", "i", "z", "redshift"]]

data_learn = np.asarray(data_learn)
x = data_learn[:, 0:5]
y = data_learn[:, 5].T
# print(x.shape)
# print(y.shape)
frst = forest.learn_forest_learn(x, y, quan, depth)

y_pred = forest.predict_forest_predict(frst, x)

plt.plot(y_pred, y, 'x')
plt.savefig('redshift.png')

# print('std ', np.std(y_pred))

std = {"std": np.std(y_pred)}
with open('redshift.json', 'w') as out:
    json.dump(std, out)

data = pd.read_csv("sdss.csv")
to_predict = np.asarray(data)
redshift_pred = forest.predict_forest_predict(frst, to_predict)

data['redshift'] = redshift_pred
data.to_csv('sdss_predict.csv')
# print(redshift_pred)
#
# plt.show()
