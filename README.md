# CoVR-VidLLM-CVPR25



## Description
This repository contains the code for CoVR-VidLLM workshop CVPR-2025.

Please visit our [Workshop Page]() for more details.

The repository structure: 

```markdown
📦 covr
 ┣ 📂 configs                 # hydra config files
 ┣ 📂 src                     # Pytorch datamodules
 ┣ 📂 tools                   # scripts and notebooks
 ┣ 📜 .gitignore
 ┣ 📜 README.md
 ┣ 📜 test.py                 # test script

 ```

## Installation

### Create environment

```bash
conda create --name covr-env
conda activate covr-env
```

To install the necessary packages, use requirements.txt file:
```bash
python -m pip install -r requirements.txt
```

The code was tested on Python 3.10 and PyTorch 2.4.


### (Optional) Download pre-trained models

To download the checkpoints, run:
```bash
bash tools/scripts/download_pretrained_models.sh
```



## Usage

### Computing BLIP embeddings

Before evaluating, you will need to compute the BLIP embeddings for the videos/images. To do so, run:
```bash
# This will compute the BLIP embeddings for the WebVid-CoVR videos. 
# Note that you can use multiple GPUs with --num_shards and --shard_id
python tools/embs/save_blip_embs_vids.py --video_dir datasets/WebVid/2M/train --todo_ids annotation/webvid-covr/webvid2m-covr_train.csv 

# This will compute the BLIP embeddings for the WebVid-CoVR-Test videos.
python tools/embs/save_blip_embs_vids.py --video_dir datasets/WebVid/8M/train --todo_ids annotation/webvid-covr/webvid8m-covr_test.csv 

# This will compute the BLIP embeddings for the WebVid-CoVR modifications text. Only needed if using the caption retrieval loss (model/loss_terms=si_ti+si_tc).
python tools/embs/save_blip_embs_txts.py annotation/webvid-covr/webvid2m-covr_train.csv datasets/WebVid/2M/blip-vid-embs-large-all
```

### Evaluating

#### Calculating Query Features

The command to calculate the query feature results for Image/Video + description:
```bash
python test.py test=webvid-covr
```

The command to calculate the query feature description for Image/Video Description only:
```bash
python test.py test=webvid-covr_text
```

The results will be saved in a torch tensor file `query_feat.pt` and `query_feat_txt_only.pt` in the output folder for Image/Video + Description and Descriptions only respectively.

#### Calculating Recalls for evaluation

To calculate the recalls for the query features results for Image/Video + description, execute the following command:
```bash
python evaluate_scores.py evaluate=webvid-covr
```

And, to calculate the recalls for the query features results for description only, execute the following command:
```bash
python evaluate_scores.py evaluate=webvid-covr_text
```

The recalls will be saved in a json file `recalls.json` and `recalls_txt_only.pt` in the output folder for Image/Video + Description and Descriptions only respectively.

The Format of the recalls.json is as following:
```json
{
  "R1": 5.26,
  "R5": 15.79,
  "R10": 47.37,
  "R50": 100.0,
  "meanR3": 22.81,
  "meanR4": 42.11,
  "annotation": "webvid8m-covr_test_new.csv"
}
```


## Acknowledgements
Based on [CoVR](https://github.com/lucas-ventura/CoVR), [BLIP](https://github.com/salesforce/BLIP/) and [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/tree/main).

