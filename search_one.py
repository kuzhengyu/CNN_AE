from get_encoder import *
from get_encoder_feature import *
from encoder_feature_match import *
from time import *

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
    # 查晚上A，找白天B
    # =================可修改参数↓==============================
    AE_path = 'checkpoint/2.3_2.h5'  #自编码器
    model_type = '2.3_2'  # AE模型+VGG使用输出层 涉及encoder命名
    data_range = [20,300] #数据库图片B   =[0,400]  ：从第1到399张
    query_index = 270 #  查询图片A的下标   =0:查询第一张图片
    # ==============不修改↓===============Alderley dataset======================
    encoder_path = './model/'+model_type+'_encoder.h5'  #encoder
    night_path ='./raw_data/FRAMESA/' # 晚上A的数据集
    day_path = './raw_data/FRAMESB/'  # 白天B的数据集
    day_feature_path = './feature/'+model_type+'_B_'+str(data_range[0])+'_'+str(data_range[1])+'.npy' #提取后特征的地址

    get_encoder(AE_path,model_type) #获取encoder，保存在model文件夹  #有可能修改了AE的，注意
    begin_time = time()
    get_one_encoder_feature(day_path,encoder_path,model_type,data_range,day_feature_path) #获取经encoder编码的白天特征向量，保存在feature文件夹下
    extract_time = time()-begin_time
    one_feature_match(encoder_path,day_feature_path,night_path,query_index,data_range) #查询第query_index+1张图片A
    matching_time = time()-begin_time

    print('单个提取时间：', extract_time/(data_range[1]-data_range[0]), 's')
    print('单个匹配时间：', matching_time/(data_range[1]-data_range[0]), 's')