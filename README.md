# ProNC
This is the official Repository for the paper, ["Rethinking Continual Learning with Progressive Neural Collapse" (ICLR 2026)](https://openreview.net/forum?id=E3bBZ02Qcc&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))

## Setup Environment
Create the environment with `conda create -n env python=3.10` <br>
Install the necessary packages wtih `pip install -r requirements.txt`

## Run the experiments
```
python utils/main.py --backbone resnet18 --model exorth_NCT --dataset seq-cifar100 --n_epochs 50 --lr 0.03 --batch_size 32 --buffer_size 200 --device 0 --num_workers 2 --optim_mom 0. --optim_wd 0.0000 --lr_scheduler multisteplr --lr_milestones 35 45 --main_weight 18 --distill_weight 170
```
