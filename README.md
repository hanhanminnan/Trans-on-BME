# Trans-on-BME
This is the code repository for our Paper.

We command you use anaconda to configure the enviroment.You can use the following code to create the enviroment.
```
conda env create -f environment.yml
```

Then you can use the following code to train the RL net(for 3agent situation).
```
CUDA_VISIBLE_DEVICES=0 nohup python DQN.py --task train --files your train_image.txt location  your train_landmark.txt --file_type brain --landmarks 0 1 2 --multiscale --viz 0 --train_freq 50 --init_memory 5000 --memory_size 18000 --batch_size 32 --save_freq 50 &
```
