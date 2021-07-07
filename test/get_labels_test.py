from celegans_dataset import get_C_elegants_label
if __name__ == '__main__':
    filename = 'E:\workspace\线虫数据集\图片整理\\6_0.417_0.500\\b_1_2_6_2.JPG'
    labels_name_required = ['part', 'batch', 'remaining_days', 'photo_id', 'shoot_days']
    l = get_C_elegants_label(filename, labels_name_required)
    print(l.shape)