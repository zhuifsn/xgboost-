#coding=utf-8
import xgboost as xgb
import pandas as pd
import time 
import numpy as np
now = time.time()
dataset = pd.read_csv("mnist_train",header=None)
#pd.read_csv("../input/train.csv") # ע���Լ�����·��
train = dataset.iloc[:,:784].values
labels = dataset.iloc[:,784:785].values
tests = pd.read_csv("mini_test.csv") # ע���Լ�����·��
#test_id = range(len(tests))
test = tests.iloc[:,:].values
test

params={
'booster':'gbtree',
# ������д������0-9����һ����������⣬��˲�����multisoft���������
'objective': 'multi:softmax', 
'num_class':10, # �������� multisoftmax ����
'gamma':0.05,  # ������Ҷ�ӽڵ���һ����������С��ʧ��Խ���㷨ģ��Խ���� ��[0:]
'max_depth':12, # ����������� [1:]
#'lambda':450,  # L2 ������Ȩ��
'subsample':0.4, # ����ѵ�����ݣ�����Ϊ0.5�����ѡ��һ�������ʵ�� (0:1]
'colsample_bytree':0.7, # ��������ʱ�Ĳ������� (0:1]
#'min_child_weight':12, # �ڵ������������
'silent':1 ,
'eta': 0.005, # ��ͬѧϰ��
'seed':710,
'nthread':4,# cpu �߳���,�����Լ�U�ĸ����ʵ�����
}

plst = list(params.items())

#Using 10000 rows for early stopping. 
offset = 50000  # ѵ����������50000������35000����ѵ����15000������֤

num_rounds = 50 # ���������
xgtest = xgb.DMatrix(test)

# ����ѵ��������֤�� 
xgtrain = xgb.DMatrix(train[:offset,:], label=labels[:offset])
xgval = xgb.DMatrix(train[offset:,:], label=labels[offset:])

# return ѵ������֤�Ĵ�����
watchlist = [(xgtrain, 'train'),(xgval, 'val')]


# training model 
# early_stopping_rounds �����õĵ��������ϴ�ʱ��early_stopping_rounds ����һ���ĵ���������׼ȷ��û��������ֹͣѵ��
model = xgb.train(plst, xgtrain, num_rounds, watchlist,early_stopping_rounds=100)
#model.save_model('./model/xgb.model') # ���ڴ洢ѵ������ģ��
preds = model.predict(xgtest,ntree_limit=model.best_iteration)

# ��Ԥ����д���ļ�����ʽ�кܶ࣬�Լ�˳����ʵ�ּ���
#np.savetxt('submission_xgb_MultiSoftmax.csv',np.c_[range(1,len(test)+1),preds],
 #               delimiter=',',header='ImageId,Label',comments='',fmt='%d')


cost_time = time.time()-now
print("end ......",'\n',"cost time:",cost_time,"(s)......")