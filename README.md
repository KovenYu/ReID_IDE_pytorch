# IDE model for Re-ID

this is a baseline model which is exactly the same as standard image classification model using softmax loss.

more specifically, every identity is recognized as a specific class during training.

for example, in Market-1501 there are 751 classes (751 identities).

here the backbone network is ResNet-50.

when deployment, extracting feature from the second last layer.

rank-1 accuracy is, around 75%, in Market-1501. don't care too much about it. just a baseline.


### usage
##### environments

- python 2.7, pytorch 0.3.0, matlab
- assuming you have a GPU

##### preliminary

- run ./data/make_imdb_Duke.m and ./data/make_imdb_Market.m in MATLAB to organize the datasets.
- don't forget to replace the *dir_path* with yours, which should contain the original dataset,
where the folders are renamed as 'train', 'test' and 'query'.
if not clear, you can read the code without difficulty.
- download the pretrained model parameter from https://download.pytorch.org/models/resnet50-19c8e357.pth
and put it into ./data/
- if still any problem, please contact me for the wrapped data.

##### running

- determine your gpu id and save path (for saving checkpoints and logs), and run
```bash
python main.py --gpu your_gpu_id --save_path your_save_path
```