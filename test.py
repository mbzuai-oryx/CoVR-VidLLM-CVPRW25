import shutil
from pathlib import Path
import torch

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig



@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig):
    fabric = instantiate(cfg.trainer.fabric)
    fabric.launch()

    model = instantiate(cfg.model)
    model = fabric.setup(model)

    for dataset in cfg.test:
        columns = shutil.get_terminal_size().columns
        fabric.print("-" * columns)
        fabric.print(f"Testing {cfg.test[dataset].dataname}".center(columns))

        data = instantiate(cfg.test[dataset])
        test_loader = fabric.setup_dataloaders(data.test_dataloader())

        test = instantiate(cfg.test[dataset].test)
        query_feats = test(model, test_loader, fabric=fabric)
        
        suffix = "_txt_only" if cfg.test[dataset].test["_target_"] == "src.test.webvid_covr_text.TestWebVidCoVRTextOnly" else ""

        # Save query features
        torch.save(query_feats, f'query_feat{suffix}.pt')
        print(f"Query Features saved in {Path.cwd()} as query_feat{suffix}.pt")

if __name__ == "__main__":
    main()
