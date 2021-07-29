import sys
sys.path.append('../')
from lib.celegans_dataset import generate_C_elegans_csv
if __name__ == '__main__':
    sys.path.append('./models/alexnet')
    from config import config
    generate_C_elegans_csv(config.root_dir, config.csv_tar_path, num_val=10, num_train=10, shuffle=True, skip_missing_shootday=True)