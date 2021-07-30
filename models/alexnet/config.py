class Celegans_dataset:
    root_dir = 'E:\workspace\线虫数据集\分类数据集/图片整理'             # 数据集根目录
    csv_tar_path = '.\csv files'                                # 包含csv文件的文件夹
    csv_file_train = '.\\csv files\cele_df_train.csv'           # 训练集csv文件路径
    csv_file_val = '.\\csv files\cele_df_val.csv'               # 测试集csv文件路径
    # 需要的标签信息,可选的值有:['part', 'batch', 'elegans_id', 'remaining_days', 'photo_id', 'shoot_days']
    labels_name_required = 'shoot_days'                         

class Config:
    model_dir = ''                                              # 使用的模型文件夹
    resume_weights = None                                       # 设置训练起始epoch
    root_dir = Celegans_dataset.root_dir                        
    csv_tar_path = Celegans_dataset.csv_tar_path
    csv_file_train = Celegans_dataset.csv_file_train
    csv_file_val = Celegans_dataset.csv_file_val
    labels_name_required = Celegans_dataset.labels_name_required
    
    train_batch_size = 32               # 训练的batch size
    val_batch_size = 8                  # 验证集的batch size
    TORCH_HOME = 'E:\workspace'         # 设置pytorch路径，用于指定预训练权重下载路径
    
    pretrained = True                   # 是否预训练
    learning_rate = 0.01                # 学习率
    num_epochs = 30                     # 迭代次数

    begin_epoch = 0                     # 起始epoch

config = Config()