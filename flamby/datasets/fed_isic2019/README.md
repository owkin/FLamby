
1)
To import the ISIC 2019 data and back out the datacenter attribution of images, run: python download_ISIC_2019_raw_data.py

2)
To create folds, run: python folds.py

3)
To preprocess and resize images, run:
python resize_images.py --input_folder ../ISIC_2019_Training_Input --output_folder ../ISIC_2019_Training_Input_preprocessed  --mantain_aspect_ratio --sz 224 --cc

This will resize all images such that the shorter side of the image is of size 224px while mantaining the aspect ratio of the image.
This kind of preprocessing was also used by ISIC 2019.
To also add color constancy, simply add --cc flag to the command.
For a complete list of parameters, run python resize_images.py -h

4)
To train the model, run:
python train.py \
--model_name efficient_net \
--arch_name efficientnet-b3 \
--device 'cpu' \
--metric 'auc' \
--training_folds_csv ./train_folds.csv \
--train_data_dir ./ISIC_2019_Training_Input_preprocessed \
--kfold 0,1,2,3,4 \
--pretrained imagenet \
--train_batch_size 64 \
--valid_batch_size 64 \
--learning_rate  5e-4 \
--epochs 1 \
--sz 100 \
--accumulation_steps 8 \
--loss 'weighted_focal_loss'


