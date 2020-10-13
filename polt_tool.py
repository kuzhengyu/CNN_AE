import pylab as pl
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(seqA_data_range,sim_path):
    data =[]
    # 读取数据
    f1 = open(sim_path, 'rb')
    for x in range(seqA_data_range[1] - seqA_data_range[0]):
        data.append(np.load(f1))
    f1.close()
    data = np.array(data)
    # 绘制热力图
    sns.heatmap(data=data,  # 指定绘图数据
                cmap='winter',  # 指定填充色
                # linewidths=0,  # 设置每个单元格边框的宽度
                # fmt='.1e', # 以科学计算法显示数据
                vmin = 0.7, # limits of the colormap
                vmax = 1
                )
    # 添加标题
    plt.title('Similarity')
    # 显示图形
    plt.show()


# 相差10帧之内，真实值为真
def truth_value(query_frame,predict_frame):
    if(abs(query_frame-predict_frame)<=10):
        return True
    else:
        return False

# 余弦值大于阈值，即预测值为真
def predict_value(consine,threshould):
    if(consine>=threshould):
        return True
    else:
        return False

def calculate_PR(seqA_data_range,result_path,threshold = 0.8):
    n = seqA_data_range[1] - seqA_data_range[0]

    # 读取数据
    data = []
    f1 = open(result_path, 'rb')
    for x in range(n):
        data.append(np.load(f1))
    f1.close()
    data = np.array(data)

    #计算
    truth = []
    predict = []
    for row in range(n):
        max_sim = sorted(data[row],reverse=True)[0]
        max_index = np.array(data[row]).argsort()[::-1][0]
        if predict_value(max_sim,threshold):
            predict.append(max_sim)
        else:
            predict.append(-1)
        if truth_value(row,max_index):
            truth.append(1)
        else:
            truth.append(-1)

    precision, recall, thresholds = precision_recall_curve(
        truth, predict)
    area = auc(recall, precision)

    pl.figure(-1)
    pl.clf()
    pl.plot(recall, precision, label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0,1.05])
    pl.xlim([0.0,1.0])
    pl.title('Precision-Recall example: AUC=%0.2f' % area)
    pl.legend(loc="lower left")
    pl.show()


