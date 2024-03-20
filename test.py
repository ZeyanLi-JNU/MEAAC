import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
from functools import partial
import torch
import os
from torch.utils.data import DataLoader
from my_dataset_coco import CocoDetection
from transforms import Compose, ToTensor
from train import *


def extract_features(train_loader, model, device):
    features = []
    all_labels = []
    with torch.no_grad():
        for imgs, targets_batch in train_loader:
            imgs = imgs.to(device)
            batch_features = model.encoder(imgs)
            batch_features = batch_features.view(batch_features.size(0), -1)
            features.extend(batch_features.cpu().numpy())
            for targets in targets_batch:
                label = targets['labels'][0].item() - 1
                all_labels.append(label)
    return np.array(features), np.array(all_labels)


def optimize_hyperparameters(model_class, param_bounds, X, y):
    def optimize_model(**params):
        params['max_depth'] = int(params['max_depth'])
        if 'n_estimators' in params:
            params['n_estimators'] = int(params['n_estimators'])

        estimator = model_class(**params)
        estimator.fit(X, y)
        preds = estimator.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)
        return auc

    optimizer = BayesianOptimization(f=optimize_model, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=2, n_iter=3)
    return optimizer.max['params']


def evaluate_model(model_class, best_params, features, labels):
    if 'max_depth' in best_params:
        best_params['max_depth'] = int(best_params['max_depth'])
    if 'n_estimators' in best_params:
        best_params['n_estimators'] = int(best_params['n_estimators'])

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = model_class(**best_params)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'AUCROC': roc_auc_score(y_test, preds_proba),
        'F1': f1_score(y_test, preds),
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds)
    }
    print(metrics)
    return metrics


def run_evaluation_and_save_results(train_loader, models, device, rf_param_bounds):
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        features, labels = extract_features(train_loader, model, device)
        best_params = optimize_hyperparameters(RandomForestClassifier, rf_param_bounds, features, labels)
        metrics = evaluate_model(RandomForestClassifier, best_params, features, labels)
        results[name] = metrics

    results_df = pd.DataFrame(results).T
    results_df.to_csv('Parameters_evaluation_metrics.csv')
    print('Metrics saved to Parameters_evaluation_metrics.csv')


transforms = Compose([
    ToTensor(),
])


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    targets = list(targets)
    return imgs, targets


data_root = './archive'
dataset_name = "test"
train_dataset = CocoDetection(root=data_root, dataset=dataset_name, transforms=transforms)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAAC = torch.load('MEAAC.pth').to(device)
MEAAC.eval()

models = {'MEAAC': MEAAC,}

rf_param_bounds = {
    'n_estimators': (10, 200),
    'max_depth': (1, 50),
}

run_evaluation_and_save_results(train_loader, models, device, rf_param_bounds)