import torch.nn as nn
import torch 
import torch.nn.functional as F
from models import WeightedFocalLoss
import types

def scale_and_map_df(df, cols):  
    if 'age_approx' in cols:
        if df['age_approx'].max() > 100:
            raise ValueError(f"DF max age_approx > 100 - {df['age_approx'].max()}")
        df['age_approx'] /= 100
    if 'sex' in cols:
        df['sex'] = df['sex'].map({
            'male':0,
            'female':1
        })
    #TODO: read ISIC 2019 paper and find how to encode missing value
    df['age_approx'] = df['age_approx'].fillna(-5)
    df['sex'] = df['sex'].fillna(-5)
    return df


def modify_model(model, args, nftrs=9):
    if args.model_name == 'efficient_net':
        num_cnn_features = model.base_model._fc.in_features
        model.meta_before = nn.Sequential(
            nn.Linear(nftrs, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 256),
            nn.ReLU(), 
            nn.Dropout(p=0.4))
        model.meta_after = nn.Sequential(
            nn.Linear(num_cnn_features + 256, 1024),
            nn.BatchNorm1d(1024), 
            nn.ReLU())
        model.base_model._fc = nn.Linear(1024, 1)
    
    def new_forward(self, image, target, meta, weights=None, args=None):
        cnn_features = self.base_model.extract_features(image) 
        cnn_features = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)
        if self.base_model._dropout:
            cnn_features = F.dropout(cnn_features, p=0.2, training=self.training)
        meta_features = self.meta_before(meta)

        if not self.training and args.tta:
            meta_features = meta_features.repeat_interleave(args.num_crops,0)

        features = torch.cat([cnn_features, meta_features], dim=1)
        features = self.meta_after(features)
        out = self.base_model._fc(features)
        if not args.loss=='weighted_bce' and weights is not None:
            weights_ = weights[target.data.view(-1).long()].view_as(target)
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_func(out, target.view(-1,1).type_as(out))
            loss_class_weighted = loss * weights_
            loss = loss_class_weighted.mean()
        elif args.loss == 'bce':
            loss = nn.BCEWithLogitsLoss()(out, target.view(-1,1).type_as(out))
        elif args.loss == 'weighted_focal_loss':
            loss = WeightedFocalLoss()(out, target.view(-1,1).type_as(out))
        elif args.loss == 'focal_loss':
            loss = FocalLoss()(out, target.view(-1,1).type_as(out))
        return out, loss

    model.forward  = types.MethodType(new_forward, model)
    return model