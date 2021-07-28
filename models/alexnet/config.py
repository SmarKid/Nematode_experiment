class Celegans_dataset:
    root_dir = 'F:/线虫分类数据集'
    csv_tar_path = 'F:/zzk files/workspace/线虫实验/csv files'  # 包含csv文件的文件夹
    csv_file_train = '.\\csv files\cele_df_train.csv'           # 训练集csv文件路径
    csv_file_val = '.\\csv files\cele_df_val.csv'               # 测试集csv文件路径
    # 需要的标签信息 ['part', 'batch', 'elegans_id', 'remaining_days', 'photo_id', 'shoot_days']
    labels_name_required = 'shoot_days'                         

class Config:
    model_dir = ''
    resume_weights = None
    root_dir = Celegans_dataset.root_dir
    csv_tar_path = Celegans_dataset.csv_tar_path
    csv_file_train = Celegans_dataset.csv_file_train
    csv_file_val = Celegans_dataset.csv_file_val
    labels_name_required = Celegans_dataset.labels_name_required
    
    train_batch_size = 32
    test_batch_size = 8
    TORCH_HOME = 'F:/zzk files'         # 设置pytorch路径，用于指定预训练权重下载路径
    
    learning_rate = 0.01
    num_epochs = 20

    begin_epoch = 0

config = Config()