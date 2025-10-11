import yaml
import torch
import argparse
import warnings, os
from time import time
from modules import CollDTI
from trainer import Trainer
from dataloader import LoadDataset
from process_data import process_data
from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader
from utils import set_seed, graph_collate_func, mkdir

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return CN(config_dict)

def utils():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description="CollDTI for DTI prediction")
    parser.add_argument('--cfg', default="configs/CollDTI.yaml", help="path to config file", type=str)
    args = parser.parse_args()
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = load_config(args.cfg)
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR)

    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    train_data_5, val_data_5, test_data_5 = process_data(data_root_path="./data/Ning/", sv_path="./SNView/SV/feature/", device=device)

    return train_data_5, val_data_5, test_data_5, cfg, device

def main():

    train_data_5, val_data_5, test_data_5, cfg, device = utils()
    for k in range(5):
        train_data = train_data_5[k]
        val_data = val_data_5[k]
        test_data = test_data_5[k]
        train_dataset = LoadDataset(1, train_data)
        val_dataset = LoadDataset(1, val_data)
        test_dataset = LoadDataset(1, test_data)

        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'drop_last': True, 'collate_fn': graph_collate_func}
        training_generator = DataLoader(train_dataset, **params)

        params['shuffle'] = False
        params['drop_last'] = False
        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)

        model = CollDTI(**cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        torch.backends.cudnn.benchmark = True
        trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, k, **cfg)

        print("___________________ {} start__________________".format(k))
        result = trainer.train()

        with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "{}_model_architecture.txt".format(k)), "w") as wf:
            wf.write(str(model))

        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
