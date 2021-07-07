from celegans_dataset import CelegansDataset
import torch
if __name__ == '__main__':
    labels = 'shoot_days'
    csv_file = 'E:\workspace\线虫数据集\图片整理\cele_df_val.csv'
    root_dir = 'E:\workspace\线虫数据集\图片整理'
    celegans_dataset = CelegansDataset(labels, csv_file, root_dir)

    batch_size = 5
    celegans_iter = torch.utils.data.DataLoader(celegans_dataset, batch_size=batch_size)

    for batch in celegans_iter:
        print(batch['image'].shape)
        print(batch['label'].shape)
        print(type(batch['label']))
        print(batch['label'])
        break