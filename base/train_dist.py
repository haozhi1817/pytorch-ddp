import yaml

from engine_dist import Base


def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    enginer = Base(cfg=cfg)

    n_epochs = enginer.train_cfg["num_epochs"]

    for epoch in range(n_epochs):
        enginer.train_one_epoch(epoch)
        enginer.valid_one_epoch(epoch)


if __name__ == "__main__":
    cfg_path = "/disk2/haozhi/tmp/code/dist_train/config.yaml"
    main(cfg_path)
