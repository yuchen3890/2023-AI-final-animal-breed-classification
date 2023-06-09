# 2023-AI-final-animal-breed-classification

## LDP-Net based Method

### Data Structure

The data structure for the source and target domains is as follows:

#### Source Domains (for training)
- [MiniImageNet](https://drive.google.com/file/d/1uxpnJ3Pmmwl-6779qiVJ5JpWwOGl48xt/view?usp=sharing): This dataset consists of images from various categories.

#### Target Domains (for testing)
- [Rabbit Breed](https://drive.google.com/file/d/1XBwGkHBAwnKVmjD4DxwBv4dgsufzMlx5/view?usp=sharing): This dataset includes images of different rabbit.
- [CUB (Caltech-UCSD Birds-200-2011)](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1): This dataset contains images of bird species.

Please ensure that you download the necessary datasets and organize them properly within the corresponding source and target domain folders before running the code.

### Training

```
python train.py --lamba1 1.0 --lamba2 0.15 --m 0.998 --seed 1111 --epoch 100 --train_n_eposide 100 --n_support 5 --source_data_path ./source_domain/miniImageNet/train  --pretrain_model_path  ./pretrain/399.tar  --save_dir path/to/directory/saving/ckpt
```
- set the "n_support" & "n_way" arguments according to your training setting.
- n_support = 5 means 5 shot, n_way = 5 means 5 way.
- "n_way" argument is 5 by default. You can change it by adding the argument. 


### Testing

```
python test.py --n_support 5 --n_way 4 --seed 1111 --current_data_path ./target_domain/rabbit_breed  --current_class 4 --test_n_eposide 600  --model_path path/to/your/model/ckpt
```
- set the "n_support" & "n_way" arguments according to your training setting.
- n_support = 5 means 5 shot, n_way = 5 means 5 way.
- set the test dataset path to the "current_data_path" argument, and modify the corresponding number of classes to "current_class".

## P>M>F based Method

Put MiniImageNet at ./minimagenet, Rabbit Breed at ./rabbit_breed, CUB (Caltech-UCSD Birds-200-2011) at ./CUB_200_2011, then run PMF.ipynb for training, testing and evaluation.
