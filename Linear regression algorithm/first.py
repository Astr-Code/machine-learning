#导入sklearn.svm.LinearSVC, train_test_split, matplotlib, numpy, pandas模块
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

#处理数据
data = pd.read_csv(r'E:\other\doing_data_science-master\doing_data_science-master\doing_data_science-master\dds_datasets\dds_datasets\dds_ch2_nyt\nyt1.csv')
X, y = data.loc[:, ["Age", "Gender", "Impressions","Signed_In"]], data.loc[:, "Clicks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=2)
reg: object = LinearSVC().fit(X_train, y_train)
# 模型评估
print("Training set score: {:.3f}".format(reg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(reg.score(X_test, y_test)))