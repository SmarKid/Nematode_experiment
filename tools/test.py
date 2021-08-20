import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from lib.celegans_dataset import CelegansDataset, get_C_elegants_label
from lib.evaluate import evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(args, config, network):
    '''
        return:
            准确率，召回率，f1-score
    '''
    net = network()
    net.to(device)
    net.eval() # 评估模式, 这会关闭dropout
    if args.resume_weights:
        model_file = os.path.join("./models/", args.model_dir, 'weights/epoch_%d.pth' % args.resume_weights)
        check_point = torch.load(model_file, map_location=device)
        net.load_state_dict(check_point)
    ex_name = os.path.splitext(args.file_path)[1]

    if ex_name in ['.csv']:
        file_path = os.path.normpath(args.file_path)
        test_set = CelegansDataset(config.labels_name_required, file_path, config.test_dir, 
                transform=config.trans['val_trans'])
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.val_batch_size)
        df = pd.read_csv(file_path)
        columns = ['file_path', 'predict_label', 'true_label', 'confidence']
        batch_count = 0
        ret = np.empty((0,4))
        score = np.empty((0, 30))
        import tqdm
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                X = batch['image']
                true_labels = batch['label']
                true_labels = np.array(true_labels)
                outputs = net(X.to(device))
                # (batch_size, num_class)
                outputs = np.array(outputs)
                score = np.concatenate((score, outputs))
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
        # evaluate(data=ret, index=(1, 2))
        ret_df = pd.DataFrame(ret, columns=columns, dtype=str)
        ret_df.to_csv('./epoch_%d_output.csv' % args.resume_weights)
        from sklearn import metrics
        y_true = ret[:, 2].tolist()
        y_true = list(map(int, y_true))
        y_pred = ret[:, 1].tolist()
        y_pred = list(map(int, y_pred))
        y_score = score.tolist()
        print('accuracy: %.2lf' % metrics.accuracy_score(y_true, y_pred))
        labels = np.array([i for i in range(30)])
        print('top-3 accuracy: %.2lf' % metrics.top_k_accuracy_score(y_true, y_score, k=3, labels=labels))
        print('top-5 accuracy: %.2lf' % metrics.top_k_accuracy_score(y_true, y_score, k=5, labels=labels))
        print('recall_score: %.2lf' % metrics.recall_score(y_true, y_pred, average='macro'))
        print('f1-score: %.2lf' % metrics.f1_score(y_true, y_pred, average='weighted'))
    else:
        print('not supposed')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None,required=True, type=int)  
    parser.add_argument('--file_path', '-f', default=None, required=True, type=str)  
    args = parser.parse_args()
    model_root_dir = os.path.join('./models/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config import config
    from network import network
    test(args, config, network)

