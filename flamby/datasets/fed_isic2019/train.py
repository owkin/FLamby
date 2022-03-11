import torch 
import numpy as np
import argparse
from model_dispatcher import MODEL_DISPATCHER
from dataset import MelonamaDataset
import pandas as pd
import albumentations
from early_stopping import EarlyStopping
from tqdm import tqdm
from average_meter import AverageMeter
import sklearn
import os
from sklearn import metrics
from datetime import date, datetime
import pytz
from pathlib import Path
import torch.nn as nn
from utils import scale_and_map_df, modify_model
from sklearn.metrics import roc_auc_score
import random 


tz = pytz.timezone('Australia/Sydney')
syd_now = datetime.now(tz)


def train_one_epoch(args, train_loader, model, optimizer, weights=None):
    if args.loss.startswith('weighted'): weights = weights.to(args.device)
    losses = AverageMeter()
    model.train()
    if args.accumulation_steps > 1: 
        print(f"Due to gradient accumulation of {args.accumulation_steps} using global batch size of {args.accumulation_steps*train_loader.batch_size}")
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=len(train_loader))
    for b_idx, data in enumerate(tk0):
        for key, value in data.items():
            data[key] = value.to(args.device)
        if args.accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()
        _, loss = model(**data, args=args, weights=weights)

        with torch.set_grad_enabled(True):
            loss.backward()
            if (b_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg)
    return losses.avg
        

def evaluate(args, valid_loader, model):
    losses = AverageMeter()
    final_preds = []
    model.eval()
    with torch.no_grad():
        tk0 = tqdm(valid_loader, total=len(valid_loader))
        for data in tk0:
            for key, value in data.items():
                data[key] = value.to(args.device)
            preds, loss = model(**data, args=args)
            if args.loss == 'crossentropy' or args.loss == 'weighted_cross_entropy': 
                preds=preds.argmax(1)
            losses.update(loss.item(), valid_loader.batch_size) 
            preds = preds.cpu().numpy()
            final_preds.extend(preds)
            tk0.set_postfix(loss=losses.avg)
    return final_preds, losses.avg


def run(fold, args):
    if args.sz: 
        print(f"Images will be resized to {args.sz}")
        args.sz = int(args.sz)

    # get training and valid data    
    df = pd.read_csv(args.training_folds_csv)
    if args.loss == 'crossentropy' and not args.isic2019:
        diag_to_ix = {v:i for i,v in enumerate(sorted(list(set(df.diagnosis))))}
        ix_to_diag = {v:i for i,v in diag_to_ix.items()}    

    if args.external_csv_path: 
        df_external = pd.read_csv(args.external_csv_path)      
    df_train = df.query(f"kfold != {fold}").reset_index(drop=True)
    df_valid = df.query(f"kfold == {fold}").reset_index(drop=True)
    print(f"Running for K-Fold {fold}; train_df: {df_train.shape}, valid_df: {df_valid.shape}")

    # calculate weights for NN loss
    weights = len(df)/df.target.value_counts().values 
    class_weights = torch.FloatTensor(weights)
    if args.loss == 'weighted_bce': 
        print(f"assigning weights {weights} to loss fn.")
    if args.loss == 'focal_loss': 
        print("Focal loss will be used for training.")
    if args.loss == 'weighted_cross_entropy': 
        print(f"assigning weights {weights} to loss fn.")
    
    # create model
    if 'efficient_net' in args.model_name:
        model = MODEL_DISPATCHER[args.model_name](pretrained=args.pretrained, arch_name=args.arch_name, 
            ce=(args.loss=='crossentropy' or args.loss == 'weighted_cross_entropy' or args.load_pretrained_2019))
    else:
        model = MODEL_DISPATCHER[args.model_name](pretrained=args.pretrained)
    
    if args.model_path is not None:
        print(f"Loading pretrained model and updating final layer from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
        nftrs = model.base_model._fc.in_features
        model.base_model._fc = nn.Linear(nftrs, 1)

    meta_array=None
    if args.use_metadata:
        # create meta array
        sex_dummy_train = pd.get_dummies(df_train['sex'])[
            ['male', 'female']]
        site_dummy_train = pd.get_dummies(df_train['anatom_site_general_challenge'])[
                ['head/neck', 'lower extremity', 'oral/genital', 'palms/soles', 'torso','upper extremity']]
        assert max(df_train.age_approx)<100
        age_train = df_train.age_approx.fillna(-5)/100
        meta_array = pd.concat([sex_dummy_train, site_dummy_train, age_train], axis=1).values
        # modify model forward       
        if args.freeze_cnn:
            model.load_state_dict(torch.load(args.model_path))
        
        # update the forward pass
        model = modify_model(model, args)
       
        # freeze cnn
        if args.freeze_cnn:
            print("\nFreezing CNN layers!\n")
            for param in model.base_model.parameters(): 
                param.requires_grad = False        

        # add external meta to meta array
        if args.external_csv_path: 
            sex_dummy_ext = pd.get_dummies(df_external['sex'])[
                ['male', 'female']]
            df_external['anatom_site_general'] = df_external.anatom_site_general.replace(
                {'anterior torso': 'torso', 'lateral torso': 'torso', 'posterior torso': 'torso'})
            site_dummy_ext = pd.get_dummies(df_external['anatom_site_general'])[
                ['head/neck', 'lower extremity', 'oral/genital', 'palms/soles', 'torso','upper extremity']]
            assert max(df_external.age_approx)<100
            age_ext = df_external.age_approx.fillna(-5)/100
            meta_array = np.concatenate([meta_array, pd.concat([sex_dummy_ext, site_dummy_ext, age_ext], axis=1).values])   
        
        assert meta_array.shape[1]==9

    model = model.to(args.device)
  
    train_aug = albumentations.Compose([
        albumentations.RandomScale(0.07),
        albumentations.Rotate(50),
        albumentations.RandomBrightnessContrast(0.15, 0.1),
        albumentations.Flip(p=0.5),
        albumentations.IAAAffine(shear=0.1),
        albumentations.RandomCrop(args.sz, args.sz) if args.sz else albumentations.NoOp(),
        albumentations.OneOf(
            [albumentations.Cutout(random.randint(1,8), 16, 16),
             albumentations.CoarseDropout(random.randint(1,8), 16, 16)]
        ),
        albumentations.Normalize(always_apply=True)
    ])

    valid_aug = albumentations.Compose([
        albumentations.CenterCrop(args.sz, args.sz) if args.sz else albumentations.NoOp(),
        albumentations.Normalize(always_apply=True),
    ])

    print(f"\nUsing train augmentations: {train_aug}\n")

    # get train and valid images & targets and add external data if required (external data only contains melonama data)    
    train_images = df_train.image.tolist() 
    if args.external_csv_path:
        external_images = df_external.image.tolist()
        if args.exclude_outliers_2019:
            # from EDA notebook
            external_images = np.load(f'/home/ubuntu/repos/kaggle/melonama/data/external/clean_external_2019_{args.sz}.npy').tolist()
        print(f"\n\n{len(external_images)} external images will be added to each training fold.")
        train_images = train_images+external_images
    if args.use_pseudo_labels:
        test_df = pd.read_csv('/home/ubuntu/repos/kaggle/melonama/data/test.csv')
        test_images = test_df.image_name.tolist()
        
        if args.pseudo_images_path:
            test_images = list(np.load(args.pseudo_images_path, allow_pickle=True))

        print(f"\n\n{len(test_images)} test images will be added to each training fold.")        
        train_images = train_images+test_images

    train_image_paths = [os.path.join(args.train_data_dir, image_name+'.jpg') for image_name in train_images]
    train_targets = df_train.target if not args.external_csv_path else np.concatenate([df_train.target.values, np.ones(len(external_images))])

    if args.use_pseudo_labels:
        train_targets = np.concatenate([train_targets, np.load(args.pseudo_labels_path, allow_pickle=True)])

    if args.loss == 'crossentropy':
        df_train['diagnosis'] = df_train.diagnosis.map(diag_to_ix)
        train_targets = df_train.diagnosis.values

    assert len(train_image_paths) == len(train_targets), "Length of train images {} doesnt match length of targets {}".format(len(train_images), len(train_targets))

    # same for valid dataframe
    valid_images = df_valid.image.tolist() 
    valid_image_paths = [os.path.join(args.train_data_dir, image_name+'.jpg') for image_name in valid_images]
    valid_targets = df_valid.target
    if args.loss == 'crossentropy':
        df_valid['diagnosis'] = df_valid.diagnosis.map(diag_to_ix)
        valid_targets = df_valid.diagnosis.values 


    print(f"\n\n Total Train images: {len(train_image_paths)}, Total val: {len(valid_image_paths)}\n\n")
    # create train and valid dataset, dont use color constancy as already preprocessed in directory
    train_dataset = MelonamaDataset(train_image_paths, train_targets, train_aug, cc=args.cc, meta_array=meta_array)
    valid_dataset = MelonamaDataset(valid_image_paths, valid_targets, valid_aug, cc=args.cc, meta_array=meta_array)

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=4)

    # create optimizer and scheduler for training 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3,5,6,7,8,9,10,11,13,15], gamma=0.5)

    es = EarlyStopping(patience=3, mode='min' if args.metric=='valid_loss' else 'max')

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(args, train_loader, model, optimizer, weights=None if not args.loss.startswith('weighted') else class_weights)
        preds, valid_loss = evaluate(args, valid_loader, model)
        predictions = np.vstack(preds).ravel()
        
        if args.loss=='crossentropy' or args.loss=='weighted_cross_entropy':
            accuracy =  metrics.accuracy_score(valid_targets, predictions)
        else:
            auc = metrics.roc_auc_score(valid_targets, predictions)

        preds_df = pd.DataFrame({'predictions': predictions, 'targets': valid_targets, 'valid_image_paths': valid_image_paths})
        print(f"Epoch: {epoch}, Train loss: {train_loss}, Valid loss: {valid_loss}, Valid Score: {locals()[f'{args.metric}']}")
        
        scheduler.step()
        for param_group in optimizer.param_groups: print(f"Current Learning Rate: {param_group['lr']}")
        dest_directory=os.path.abspath( os.path.dirname( __file__ ) )
        es(
            locals()[f"{args.metric}"], model, 
            model_path=f"{dest_directory}/models/{syd_now.strftime(r'%d%m%y')}/{args.arch_name}_fold_{fold}_{args.sz}_{locals()[f'{args.metric}']}.bin",
            preds_df=preds_df, 
            df_path=f"{dest_directory}/valid_preds/{syd_now.strftime(r'%d%m%y')}/{args.arch_name}_fold_{fold}_{args.sz}_{locals()[f'{args.metric}']}.bin",
            args=args
            )
        if es.early_stop:
            return preds_df


def main():
    parser = argparse.ArgumentParser()
    # Required paramaters
    parser.add_argument(
        "--device", 
        default=None, 
        type=str, 
        required=True, 
        help="device on which to run the training"
    )
    parser.add_argument(
        '--training_folds_csv', 
        default=None, 
        type=str, 
        required=True, 
        help="training file with Kfolds"
    )
    parser.add_argument(
        '--model_name', 
        default='se_resnext_50',
        type=str, 
        required=True, 
        help="Name selected in the list: " + f"{','.join(MODEL_DISPATCHER.keys())}"
    )
    parser.add_argument(
        '--train_data_dir', 
        required=True, 
        help="Path to train data files."
    )
    parser.add_argument(
        '--kfold', 
        required=True,
        help="Fold for which to run training and validation."
    )
    #Other parameters
    parser.add_argument('--metric', default='auc', help="Metric to use for early stopping and scheduler.")
    parser.add_argument('--pretrained', default=None, type=str, help="Set to 'imagenet' to load pretrained weights.")
    parser.add_argument('--train_batch_size', default=64, type=int, help="Training batch size.")
    parser.add_argument('--valid_batch_size', default=32, type=int, help="Validation batch size.")
    parser.add_argument('--learning_rate', default=1e-4, type=float, help="Learning rate.")
    parser.add_argument('--epochs', default=3, type=int, help="Num epochs.")
    parser.add_argument('--accumulation_steps', default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument('--sz', default=None, type=int, help="The size to which RandomCrop and CenterCrop images.")
    parser.add_argument('--loss', default='weighted_focal_loss', help="loss fn to train")
    parser.add_argument('--external_csv_path', default=False, type=str, help="External csv path with melonama image names.")
    parser.add_argument('--cc', default=False, action='store_true', help="Whether to use color constancy or not.")
    parser.add_argument('--arch_name', default='efficientnet-b0', help="EfficientNet architecture to use for training.")
    parser.add_argument('--use_metadata', default=False, action='store_true', help="Whether to use metadata")
    parser.add_argument('--tta', default=False, action='store_true')
    parser.add_argument('--freeze_cnn', default=False, action='store_true')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--isic2019', default=False, action='store_true')
    parser.add_argument('--load_pretrained_2019', default=False, action='store_true')
    parser.add_argument('--exclude_outliers_2019', default=False, action='store_true')
    parser.add_argument('--use_pseudo_labels', default=False, action='store_true')
    parser.add_argument('--pseudo_labels_path')
    parser.add_argument('--pseudo_images_path')

    args = parser.parse_args()
    # if args.sz, then print message and convert to int
    kfolds = list(map(int, args.kfold.split(',')))
    if len(kfolds)>1:
        oof_df = pd.DataFrame()
        for fold in kfolds:
            print(f'\n\n {"-"*50} \n\n')
            preds_df = run(fold, args)
            oof_df = pd.concat([oof_df, preds_df])
    else: 
        oof_df = run(kfolds[0], args)

    dest_directory=os.path.abspath( os.path.dirname( __file__ ) )
    oof_df.to_csv(f"{dest_directory}/models/{syd_now.strftime(r'%d%m%y')}/{args.model_name}_{args.sz}_oof.csv", index=False)
    print(f"oof_df saved to {dest_directory}/models/{syd_now.strftime(r'%d%m%y')}/{args.model_name}_{args.sz}_oof.csv")

    print(f'\n\n OOF AUC: {roc_auc_score(oof_df.targets, oof_df.predictions)}')


if __name__=='__main__':
    main()



