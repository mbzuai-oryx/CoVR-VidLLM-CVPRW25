<!-- ![image](https://github.com/user-attachments/assets/a6a1f842-9748-4cff-a3d2-cd5a9d4281b4) -->

# CoVR-VidLLM-CVPR25


## Description
This repository contains the code for CoVR-VidLLM workshop CVPR-2025.

Please visit our [Workshop Page](https://www.crcv.ucf.edu/cvpr2025-vidllms-workshop/challenges.html) for more details.

The repository structure: 

```markdown
📦 covr
 ┣ 📂 configs                 # hydra config files
 ┣ 📂 src                     # Pytorch datamodules
 ┣ 📂 tools                   # scripts and notebooks
 ┣ 📜 .gitignore
 ┣ 📜 README.md
 ┣ 📜 test.py                 # test script
 ┣ 📜 validation_set.csv      # textual part of the validation set for the challenge (should be used during the Validation phase)
 ┣ 📜 test_set.csv            # textual part of the test set for the challenge (will be published during the Test phase)




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

### Download the dataset

You can download the exact video-text triplets used in our validation and test sets from the [HuggingFace page](https://huggingface.co/datasets/omkarthawakar/CoVR-VidLLM-CVPR25).

Optionally, in case you already have the original WebVid-CoVR test dataset split you can simply provide its path because the video-text triplets were sampled from there.



## Usage

### Computing BLIP embeddings

Before evaluating, you will need to compute the BLIP embeddings for the videos. To do so, run:
```bash
# This will compute the BLIP embeddings for the WebVid-CoVR videos. 
# Note that you can use multiple GPUs with --num_shards and --shard_id

# For the validation set:
python tools/embs/save_blip_embs_vids.py --video_dir datasets/WebVid/8M/train --todo_ids validation_set.csv
# Then change the name of the generated folder to "blip-vid-embs-large-all_ours_val"

# For the test set (once it is public):
python tools/embs/save_blip_embs_vids.py --video_dir datasets/WebVid/8M/train --todo_ids test_set.csv
# Then change the name of the generated folder to "blip-vid-embs-large-all_ours_test"
```


If you are interested in using the whole WebVid-CoVR dataset (not required for the challenge):
```bash
# This will compute the BLIP embeddings for the WebVid-CoVR-Train videos.
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
# On the challenge splits:
python test.py test=webvid-covr_our_val
python test.py test=webvid-covr_our_test
```

Extra options (not required for the challenge):
```bash
# On the original WebVid-CoVR test set:
python test.py test=webvid-covr

# On description only:
python test.py test=webvid-covr_text
```

The results will be saved in a torch tensor file `query_feat.pt` and `query_feat_txt_only.pt` in the output folder for Image/Video + Description and Descriptions only respectively.

Next, make sure to fuse/average the embeddings for each video in the resulting .pth file (which is originally a (15*1000)x256 tensor, where 15 - number of key frames, 1000 - number of samples, and 256 - feature dimension).

Finally, the fused embeddings should be saved as a numpy .npy file (which should contain a 1000x256 numpy nd-array, where 1000 - number of samples and 256 - feature dimension).

Simply submit this .npy file to the evaluation server on the evalAI [challenge page](https://eval.ai/web/challenges/challenge-page/2443/overview).


#### Calculating Recalls for evaluation (not required for the challenge)
This option can be used once the labels are published, but before that use our evaluation server on the evalAI [challenge page](https://eval.ai/web/challenges/challenge-page/2443/overview).

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

