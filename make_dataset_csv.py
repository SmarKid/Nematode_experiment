
from celegans_dataset import generate_C_elegans_csv
if __name__ == '__main__':
    from config import config
    generate_C_elegans_csv(config.root_dir, config.csv_tar_path, num_val=600, shuffle=True, skip_missing_shootday=True)