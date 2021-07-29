import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from lib.celegans_dataset import CelegansDataset, get_C_elegants_label
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(args, config, Network):
    net = Network()
    net.to(device)
    net.eval() # 评估模式, 这会关闭dropout
    if args.resume_weights:
        model_file = os.path.join("./models/", args.model_dir, 'weights/epoch_%d.pth' % args.resume_weights)
        check_point = torch.load(model_file, map_location=device)
        net.load_state_dict(check_point['state_dict'])
    ex_name = os.path.splitext(args.file_path)[1]
    trans = transforms.Compose([
        transforms.Resize(size=(400, 600))
    ])
    if ex_name in ['.JPG']:
        file_path = os.path.normpath(args.file_path)
        img_PIL = Image.open(file_path)
        label = get_C_elegants_label(args.file_path, config.labels_name_required)
        trans = transforms.Compose([
            trans,
            transforms.ToTensor()
        ])
        image = trans(img_PIL)
        image = torch.unsqueeze(image, 0)

        output = net(image.to(device))
        infer_label = output.argmax(axis=1).item()
        print('预测标签为: %d' % infer_label)
        print('真实标签为: %d' % label)
        print('预测置信度为: %lf' % output[0, infer_label])

    elif ex_name in ['.csv']:
        file_path = os.path.normpath(args.file_path)
        test_set = CelegansDataset(config.labels_name_required, file_path, config.root_dir, transform=trans)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.val_batch_size)
        df = pd.read_csv(file_path)
        columns = ['file_path', 'predict_label', 'true_label', 'confidence']
        batch_count = 0
        ret = np.empty((0,4))
        import tqdm
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                X = batch['image']
                true_labels = batch['label']
                true_labels = np.array(true_labels)
                outputs = net(X.to(device))
                outputs = np.array(outputs)
                predict_labels = outputs.argmax(axis=1)
                predict_labels = np.array(predict_labels)
                confidence = outputs[range(len(outputs)),  predict_labels]
                start = batch_count * config.val_batch_size
                end = start + config.val_batch_size
                end = end if end < len(df) else len(df)
                file_paths = df.iloc[start:end, 1]
                file_paths = np.array(file_paths)
                batch_count += 1
                batch_return = np.stack((file_paths, predict_labels, true_labels, confidence), axis=1)
                ret = np.concatenate((ret, batch_return))
        ret_df = pd.DataFrame(ret, columns=columns, dtype=str)
        ret_df.to_csv('./output.csv')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None,required=True, type=int)  
    parser.add_argument('--file_path', '-f', default=None, required=True, type=str)  
    args = parser.parse_args()
    model_root_dir = os.path.join('./models/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config import config
    from network import Network
    inference(args, config, Network)

