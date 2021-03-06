# RL Project
UE18CS400SH

1. Sumukh Aithal K
2. Dhruva Kashyap

## States and Actions
States is a tensor of shape: [?, 3, 210, 160].\
Action is a tensor of shape: [?, 9].


## Get Started
```
pip install -r requirements.txt
```
Download the Enduro ROM from this [link](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html).
```
ale-import-roms <path to the ROM>/
```
## How to run:
### To record your own video
```
python play.py --record --store_path <path to save the video> --trial_name <name of the trial>
```
### To Train:
Download our dataset from this [link](https://drive.google.com/drive/folders/1htqUDA1bmSQpJXIZFqcGUoAd4ubQkHOD?usp=sharing).

```
python main.py --batch_size=64 --store_path="../trials/" --trial_names t1 t2 t3 --train --model Simple --epochs=25 --num_workers=2 --train_run_name="resnet_train"
```

### To see the model play:
Download the weights of our best model from this [link](https://drive.google.com/drive/folders/1a6GFLtML6oousfi_xwAiBGzwYHpWhAAL?usp=sharing).

```
python main.py --model ResNet --model_path models/ --train_run_name test/ --play 
```
