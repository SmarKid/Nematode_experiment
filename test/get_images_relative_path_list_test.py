from collections import OrderedDict
from lib.celegans_dataset import get_images_relative_path_list
from lib.celegans_dataset import get_C_elegants_label

if __name__ == '__main__':
    DATAPATH = 'E:\workspace\线虫数据集'
    skip_missing = True
    l = get_images_relative_path_list(DATAPATH, skip_missing_shootday=False)
    max = -1
    labels_name_required = 'shoot_days'
    cnt = {}
    for path in l:
        shoot_day = get_C_elegants_label(path, labels_name_required)
        if shoot_day not in cnt:
            cnt[shoot_day] = 1
        else:
            cnt[shoot_day] += 1
    after = dict(sorted(cnt.items(), key=lambda e: e[0]))
    for k, v in after.items():
        print(k, v)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()  
    ax.plot(after.keys(), after.values(), label='num of sample')  
    ax.set_xlabel('key')  
    ax.set_ylabel('num')  
    ax.set_title("num of sample")  
    ax.legend() 
    plt.savefig('sample distribution.jpg')