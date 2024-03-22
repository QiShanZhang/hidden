hidden
===============
This repository contains our implementation of the paper published in the workshop, "". This work produced state-of-the-art result on more challenging ASVspoof 2021 LA and DF database.

[Paper link here]()


## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
$ conda create -n hidden python=3.7
$ conda activate hidden
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
(This fairseq folder can also be downloaded from https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)
$ pip install --editable ./
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are performed on the logical access (LA) and deepfake (DF) partition of the ASVspoof 2021 dataset (train on 2019 LA training and evaluate on 2021 LA and DF evaluation database).

The ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

The ASVspoof 2021 database is released on the zenodo site.

LA [here](https://zenodo.org/record/4837263#.YnDIinYzZhE)

DF [here](https://zenodo.org/record/4835108#.YnDIb3YzZhE)

For ASVspoof 2021 dataset keys (labels) and metadata are available [here](https://www.asvspoof.org/index2021.html)

## Pre-trained wav2vec 2.0 XLSR (300M)
Download the XLSR models from [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

### Training LA
To train the model run:
```
CUDA_VISIBLE_DEVICES=0 main_SSL_LA.py --track=LA --lr=0.000001 --batch_size=14 --loss=WCE  
--track=DF --lr=0.000001 --batch_size=5 --loss=WCE  --algo=3 --comment=mymodel20ablation2
```
### Testing LA and DF

To evaluate your own model on LA and DF evaluation dataset:
```
CUDA_VISIBLE_DEVICES=0 python main_SSL_LA.py --track=LA --is_eval --eval --model_path='models/model_DF_WCE_100_5_1e-06_mymodel20ablation31/epoch_2.pth' --eval_output='models/model_DF_WCE_100_5_1e-06_mymodel20ablation31/eval_CM_scores_file_SSL_LAepoch2.txt'

CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py --track=DF --is_eval --eval --model_path='/path/to/your/best_SSL_model_LA.pth' --eval_output='eval_CM_scores_file_SSL_DF.txt'
```

We also provide a pre-trained models. To use it you can run: 

Pre-trained SSL antispoofing models are available for LA and DF [here](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB?usp=sharing)

```
CUDA_VISIBLE_DEVICES=0 python main_SSL_LA.py --track=LA --is_eval --eval --model_path='/path/to/Pre_trained_models/best_SSL_model_LA.pth' --eval_output='eval_pre_trained_model_CM_scores_file_SSL_LA.txt'

CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py --track=DF --is_eval --eval --model_path='/path/to/Pre_trained_models/best_SSL_model_DF.pth' --eval_output='eval_pre_trained_model_CM_scores_file_SSL_DF.txt'
```
## Results using pre-trained model:
EER: 0.82%, min t-DCF: 0.2066  on ASVspoof 2021 LA track.

EER: 2.85 % on ASVspoof 2021 DF track.

Compute the min t-DCF and EER(%) on 2021 LA and DF evaluation dataset
```
python evaluate_2021_LA.py Score_LA.txt ./keys eval

python evaluate_2021_DF.py Score_DF.txt ./keys eval
``` 
## Contact
For any query regarding this repository, please contact:

## Citation
If you use this code in your research please use the following citation:
```bibtex

```

