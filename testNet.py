from get_encoder import *
from get_encoder_feature import *
from encoder_feature_match import *
from time import *
from polt_tool import *

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
    # 两个序列进行匹配
    # =================可修改参数↓==============================
    AE_path = 'checkpoint/2.3_2.h5'  #自编码器
    model_type = '2.3_2'  # AE模型+VGG使用输出层 涉及encoder命名
    seqA_data_range = [20,300] # 序列A的图片范围   =[0,400]  ：从第1到399张
    seqB_data_range =  [20,300] # 序列B的图片范围
    testSet_name = 'Alderley'
    seqA_path ='./raw_data/FRAMESA/' # 序列A地址
    seqB_path = './raw_data/FRAMESB/'  # 序列B地址
    # ==============不修改↓===============================
    encoder_path = './model/'+model_type+'_encoder.h5'  #encoder
    seqA_feature_path = './feature/'+model_type+'_'+testSet_name+'_A'+'_'+str(seqA_data_range[0])+'_'+str(seqA_data_range[1])+'.npy' #提取后特征的地址
    seqB_feature_path = './feature/'+model_type+'_'+testSet_name+'_B'+'_'+str(seqB_data_range[0])+'_'+str(seqB_data_range[1])+'.npy'
    # 混淆矩阵保存地址
    sim_path = "./result/"+model_type+'_'+testSet_name+'_'+str(seqA_data_range[0])+'_'+str(seqA_data_range[1])+'.npy'
    # ================================================================
    #选择功能
    flag =3
    if(flag==1):
        # 提取功能
        get_encoder(AE_path,model_type) #获取encoder，保存在model文件夹  #有可能修改了AE的，注意
        # begin_time = time()
        get_encoder_feature(seqA_path,seqB_path,encoder_path,model_type,seqA_data_range,seqB_data_range,seqA_feature_path,seqB_feature_path) #获取经encoder编码的特征向量，保存在feature文件夹下
        # extract_time = time()-begin_time
        get_sim_matrix(seqA_feature_path,seqB_feature_path,seqA_data_range,seqB_data_range,sim_path) #匹配
        # matching_time = time()-begin_time
    elif flag==2 :
        # 绘画热力图
        plot_heatmap(seqA_data_range,sim_path)
    elif flag==3:
        # 查询并绘画PR图
        calculate_PR(seqA_data_range,sim_path)

    #以下时间不准确，包含了保存的步骤的时间
    # num = seqA_data_range[1] - seqA_data_range[0] + seqB_data_range[1]-seqB_data_range[0]
    # print('单个提取时间：', extract_time/(num), 's')
    # print('单个匹配时间：', matching_time/(num), 's')