import argparse

class = Train_config:
    # size
    world_size = 0
    mini_batch_size = 0
    iter_per_epoch = 0
    total_epoch = 0
    # leanring
    warm_iter = 0
    learing_rate = 0
    momentum = 0
    weight_decay = 0
    lr_decay = []
    # model
    log_dump_interval = 0
    resume_weights = None
    init_weights = None
    model_dir = ''
    log_path = ''

def run_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None,type=int)
    args = parser.parse_args()



if __name__ == '__main__':
    run_train()