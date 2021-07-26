import sys
sys.path.append('E:\workspace\python\线虫实验')
from celegans_dataset import get_images_relative_path_list
from celegans_dataset import get_C_elegants_label

if __name__ == '__main__':
    DATAPATH = 'E:\workspace\线虫数据集\分类数据集'
    l = get_images_relative_path_list(DATAPATH, skip_missing_shootday=True)
    max = -1
    labels_name_required = 'shoot_days'
    for path in l:
        shoot_day = get_C_elegants_label(path, labels_name_required)
        if shoot_day > max:
            max = shoot_day
    print(max)
    print(len(l))