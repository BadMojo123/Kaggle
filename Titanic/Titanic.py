import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from DataExploration import DataExploration

full_data = pd.read_csv('train.csv')

# DataExploration(full_data)

train_data = full_data.drop(labels='Survived', axis =1)
train_labels = full_data['Survived']

# full_data['Survived'].hist()
# plt.show()

X_data = train_data

_mean_value = X_data['Age'].mean()
X_data['Age'].fillna(value=_mean_value, inplace=True)

X_data['NormAge'] = X_data['Age']/82
X_data = X_data.drop(labels='Age', axis=1)

one_hot = pd.get_dummies(X_data['Pclass'],prefix='Pclass',drop_first=True)
X_data = X_data.drop('Pclass',axis = 1)
X_data = X_data.join(one_hot)

one_hot = pd.get_dummies(X_data['Sex'])
X_data = X_data.drop('Sex',axis = 1)
X_data = X_data.join(one_hot)

X_data['NormSibSp'] = X_data['SibSp']
X_data = X_data.drop(labels='SibSp', axis=1)

X_data['NormParch'] = X_data['Parch']/6
X_data = X_data.drop(labels='Parch', axis=1)

X_data['NormFare'] = X_data['Fare']/512
X_data = X_data.drop(labels='Fare', axis=1)

one_hot = pd.get_dummies(X_data['Embarked'],prefix='Embarked',drop_first=True)
X_data = X_data.drop('Embarked',axis = 1)
X_data = X_data.join(one_hot)

X_data.set_index('PassengerId')
X_data = X_data.drop(labels='PassengerId', axis=1)
X_data = X_data.drop(labels='Ticket', axis=1)
X_data = X_data.drop(labels='Name', axis=1)
X_data = X_data.drop(labels='Cabin', axis=1)


pd.set_option("display.max.columns", None)
print(X_data)
print()



# logsk = LogisticRegression(C=1e9)
# logsk.fit(X_data, train_labels)
# pred = logsk.predict(X_data)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4))
model.add(tf.keras.layers.Dense(4,activation=tf.keras.activations.tanh))
model.add(tf.keras.layers.Dense(4,activation=tf.keras.activations.tanh))
model.add(tf.keras.layers.Dense(4,activation=tf.keras.activations.tanh))
model.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

tf.keras.losses.MeanAbsoluteError()
history = model.fit(X_data,train_labels, epochs=150)

val = model.evaluate(X_data,train_labels)
print(val)
pred = model.predict(X_data)
pred = np.round(pred)

confusion = metrics.confusion_matrix(train_labels, pred)
print(confusion)
accuracy = (confusion[0,0] + confusion[1,1])/sum(sum(confusion))
print(accuracy)

test = pd.read_csv('test.csv')
X_data = test

_mean_value = X_data['Age'].mean()
X_data['Age'].fillna(value=_mean_value, inplace=True)

X_data['NormAge'] = X_data['Age']/82
X_data = X_data.drop(labels='Age', axis=1)

one_hot = pd.get_dummies(X_data['Pclass'],prefix='Pclass',drop_first=True)
X_data = X_data.drop('Pclass',axis = 1)
X_data = X_data.join(one_hot)

one_hot = pd.get_dummies(X_data['Sex'])
X_data = X_data.drop('Sex',axis = 1)
X_data = X_data.join(one_hot)

X_data['NormSibSp'] = X_data['SibSp']
X_data = X_data.drop(labels='SibSp', axis=1)

X_data['NormParch'] = X_data['Parch']/6
X_data = X_data.drop(labels='Parch', axis=1)

X_data['NormFare'] = X_data['Fare']/512
X_data = X_data.drop(labels='Fare', axis=1)

one_hot = pd.get_dummies(X_data['Embarked'],prefix='Embarked',drop_first=True)
X_data = X_data.drop('Embarked',axis = 1)
X_data = X_data.join(one_hot)

X_data.set_index('PassengerId')
X_data = X_data.drop(labels='PassengerId', axis=1)
X_data = X_data.drop(labels='Ticket', axis=1)
X_data = X_data.drop(labels='Name', axis=1)
X_data = X_data.drop(labels='Cabin', axis=1)

pred = model.predict(X_data)
pred = np.round(pred)
test['Survived'] = pred
print(test[['PassengerId','Survived']])
test[['PassengerId','Survived']].to_csv('result.csv',index=False)


print('END')