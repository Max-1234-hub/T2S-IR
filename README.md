## T2S-IR

This repository is an official PyTorch implementation of the paper "__A Teacher-to-Student Information Recovery Method Towards Energy-efficient Animal Activity Recognition with Low Sampling Rates__".

## Requirements

This is my experiment eviroument
- python3.7
- pytorch+cuda11.4

## Details
### 1. Original datasets
Two public datasets are used in this study, including horse-dataset and goat-dataset.
1). The horse-dataset are collected from six horses and six activities are included. It is avaliable at
https://doi.org/10.4121/uuid:2e08745c-4178-4183-8551-f248c992cb14. The reference is (Kamminga, J. W., Janßen, L. M., Meratnia, N., & Havinga, P. J. (2019). Horsing Around—A Dataset Comprising Horse Movement. Data, 4(4), 131.).
2). The goat-dataset are collected from five goats and five activities are included. It is avaliable at
https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:78937. The reference is (Kamminga, J. W., Le, D. V., Meijers, J. P., Bisby, H., Meratnia, N., & Havinga, P. J. (2018). Robust sensor-orientation-independent feature selection for animal activity recognition on collar tags. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2(1), 1-27.).

### 2. Processed data
The __"data"__ folder contains the normalized horse data and goat data. 
The horse data have been divided into training, validation, and testing data circularly according to six-fold cross-validation.
The goat data have been divided into training, validation, and testing data circularly according to five-fold cross-validation.

### 3. Trained teacher network
The __"trained_teacher_net"__ folder contains the trained teacher models of the horse-dataset and goat-dataset. 
The teacher model of the horse-dataset is trained under sampling rate of 25 Hz.
The teacher model of the horse-dataset is trained under sampling rate of 100 Hz.

### 4. Trained reconstruction network
The __"trained_reconstruction_net"__ folder contains the trained reconstruction network of the horse-dataset and goat-dataset.

### 5. Train the student model
I trained the student model by using the code inside the __"training_script.sh"__.

Here, we also post the general code in the following (taking the horse dataset as an example).
```ruby
python train_comb.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.1 --gamma 0.5 --beta 0.9999 --gpu 1 --n_skip 8 --ir_loss_weight lambda_1 --rec_loss_weight lambda_2 --trained_teacher_net 'trained_teacher_net/canet-best_25Hz_baseline_sam_fold1.pth' --trained_recon_net 'trained_reconstruction_net/reconnet-best_fold1.pth' --data_path 'data/myTensor_1.pt' --save_path 'combination_setting1'
python train_comb.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.1 --gamma 0.5 --beta 0.9999 --gpu 1 --n_skip 8 --ir_loss_weight lambda_1 --rec_loss_weight lambda_2 --trained_teacher_net 'trained_teacher_net/canet-best_25Hz_baseline_sam_fold2.pth' --trained_recon_net 'trained_reconstruction_net/reconnet-best_fold2.pth' --data_path 'data/myTensor_2.pt' --save_path 'combination_setting2'
python train_comb.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.1 --gamma 0.5 --beta 0.9999 --gpu 1 --n_skip 8 --ir_loss_weight lambda_1 --rec_loss_weight lambda_2 --trained_teacher_net 'trained_teacher_net/canet-best_25Hz_baseline_sam_fold3.pth' --trained_recon_net 'trained_reconstruction_net/reconnet-best_fold3.pth' --data_path 'data/myTensor_3.pt' --save_path 'combination_setting3'
python train_comb.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.1 --gamma 0.5 --beta 0.9999 --gpu 1 --n_skip 8 --ir_loss_weight lambda_1 --rec_loss_weight lambda_2 --trained_teacher_net 'trained_teacher_net/canet-best_25Hz_baseline_sam_fold4.pth' --trained_recon_net 'trained_reconstruction_net/reconnet-best_fold4.pth' --data_path 'data/myTensor_4.pt' --save_path 'combination_setting4'
python train_comb.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.1 --gamma 0.5 --beta 0.9999 --gpu 1 --n_skip 8 --ir_loss_weight lambda_1 --rec_loss_weight lambda_2 --trained_teacher_net 'trained_teacher_net/canet-best_25Hz_baseline_sam_fold5.pth' --trained_recon_net 'trained_reconstruction_net/reconnet-best_fold5.pth' --data_path 'data/myTensor_5.pt' --save_path 'combination_setting5'
python train_comb.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.1 --gamma 0.5 --beta 0.9999 --gpu 1 --n_skip 8 --ir_loss_weight lambda_1 --rec_loss_weight lambda_2 --trained_teacher_net 'trained_teacher_net/canet-best_25Hz_baseline_sam_fold6.pth' --trained_recon_net 'trained_reconstruction_net/reconnet-best_fold6.pth' --data_path 'data/myTensor_6.pt' --save_path 'combination_setting6'
```
