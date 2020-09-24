import os

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        for name in dirs:
            print(os.path.join(root, name))

        # root所指的是当前正在遍历的这个文件夹的本身的地址
        # print(dirs) #当前路径下所有子目录
        # print(files) #当前路径下所有非目录子文件

if __name__ == '__main__':
    # file_name('./raw_data/Place/data_256/d')
    print(os.listdir('./raw_data/Place/data_256/m'))
    # 先处理到i