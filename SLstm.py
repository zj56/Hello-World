import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.recurrent import LSTM
from keras.layers import LSTM
#输出结果
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
from keras.layers import Bidirectional
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
#导入必要的库
#读取数据，0对应第一支股票，1对应第二只，以此类推
df1=pd.read_csv('AAPL.csv')

feanum=4#一共有多少特征
window=10#时间窗设置
df1=df1.iloc[0:,1:]#选取从第3600行开始的数据 大概是2006年一月
df1=df1.iloc[0:,0:4]
# print(df1.tail())
# scaler = MinMaxScaler()
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)
print(df.tail())
#这一部分在处理数据 将原始数据改造为LSTM网络的输入
stock=df
seq_len=window
amount_of_features = len(stock.columns)#有几列
data = stock.values #pd.DataFrame(stock) 表格转化为矩阵
sequence_length = seq_len + 1#序列长度+1
result = []
for index in range(len(data) - sequence_length):#循环 数据长度-时间窗长度 次
    result.append(data[index: index + sequence_length])#第i行到i+5
result = np.array(result)#得到样本，样本形式为 window*feanum
cut=628#分训练集测试集 最后cut个样本为测试集
train = result[:-cut, :]
x_train = train[:, :-1]
y_train = train[:, -1][:,0]
x_test = result[-cut:, :-1]
y_test = result[-cut:, -1][:,0]
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
#展示下训练集测试集的形状 看有没有问题
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
#建立、训练模型过程
d = 0.5
model = Sequential()#建立层次模型
# model.add(LSTM(64, input_shape=(window, feanum), return_sequences=True))#建立LSTM层
# model.add(Dropout(d))#建立的遗忘层
# model.add(LSTM(16, input_shape=(window, feanum), return_sequences=False))#建立LSTM层
# model.add(Dropout(d))#建立的遗忘层
# model.add(Dense(4,kernel_initializer='uniform',activation='relu'))   #建立全连接层
# model.add(Dense(1,kernel_initializer='uniform',activation='relu'))
model.add(Bidirectional(LSTM(64, input_shape=(window, feanum), return_sequences=True)))
model.add(Dropout(d))#建立的遗忘层
model.add(Bidirectional(LSTM(16, input_shape=(window, feanum), return_sequences=False)))
model.add(Dropout(d))
model.add(Dense(4,kernel_initializer='uniform',activation='relu'))   #建立全连接层
model.add(Dense(1,kernel_initializer='uniform',activation='relu'))
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs =100, batch_size = 256) #训练模型nb_epoch次
model.summary()#总结模型
#在训练集上的拟合结果
y_train_predict=model.predict(X_train)[:,0]
y_train=y_train
draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
draw.iloc[0:1885,0].plot(figsize=(12,6))
draw.iloc[0:1885,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Bidirectional LSTM Train Data",fontsize='30') #添加标题
plt.show()
#展示在训练集上的表现


#在测试集上的预测
y_test_predict=model.predict(X_test)[:,0]
y_test=y_test
draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))
draw.iloc[:,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Bidirectional LSTM Test Data",fontsize='30') #添加标题
plt.show()
# pred_test_plot= min_max_scaler.inverse_transform(y_test_predict)
# ts_data=min_max_scaler.inverse_transform(y_test)
# # 标签
# plt.plot(ts_data,label='actual value')
# plt.plot(pred_test_plot, "-.",label='estimate')
# plt.legend()
# plt.show()
#展示在测试集上的表现
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
print('训练集上的MAE/MSE/MAPE')
print(mean_absolute_error(y_train_predict, y_train))
print(mean_squared_error(y_train_predict, y_train) )
print(mape(y_train_predict, y_train) )
print('测试集上的MAE/MSE/MAPE')
print(mean_absolute_error(y_test_predict, y_test))
print(mean_squared_error(y_test_predict, y_test) )
print(mape(y_test_predict, y_test) )
y_var_test=y_test[1:]-y_test[:len(y_test)-1]
y_var_predict=y_test_predict[1:]-y_test_predict[:len(y_test_predict)-1]
txt=np.zeros(len(y_var_test))
for i in range(len(y_var_test-1)):
    txt[i]=np.sign(y_var_test[i])==np.sign(y_var_predict[i])
result=sum(txt)/len(txt)
print('预测涨跌正确:',result)