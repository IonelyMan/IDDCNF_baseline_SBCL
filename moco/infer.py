"""Evaluate a trained SBCL linear classifier on data/val/{benign,mal}.

Example:
    python moco/infer.py --data /data/malware_images \
        --checkpoint runs/binary_linear/liner_checkpoint.pth.tar

ACSA is the mean of benign and mal recall; GM is their geometric mean.
PR AUC is the trapezoidal area under the precision-recall curve for mal.
"""

import argparse
import os

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from imagenet_lt_loader import ImageNetLT_val
from utils import NormedLinear_Classifier


CLASS_NAMES = ['benign', 'mal']


def load_classifier(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint or 'arch' not in checkpoint:
        raise ValueError('Expected a checkpoint from moco/linear_classify.py with state_dict and arch.')

    arch = checkpoint['arch']
    if arch not in models.__dict__ or not callable(models.__dict__[arch]):
        raise ValueError(f'Unknown torchvision architecture in checkpoint: {arch}')
    model = models.__dict__[arch](num_classes=2)
    state = {key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}
    if 'fc.weight' not in state:
        raise ValueError('Checkpoint has no classifier weights; use the second-stage linear checkpoint.')

    feature_dim = model.fc.in_features
    fc_shape = tuple(state['fc.weight'].shape)
    if fc_shape == (feature_dim, 2):
        model.fc = NormedLinear_Classifier(num_classes=2, feat_dim=feature_dim)
    elif fc_shape != (2, feature_dim):
        raise ValueError(f'Expected a two-class classifier, got fc.weight shape {fc_shape}.')

    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def calculate_metrics(y_true, y_pred, mal_score):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mal_score = np.asarray(mal_score)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    benign_recall = tn / (tn + fp) if tn + fp else float('nan')
    mal_recall = tp / (tp + fn) if tp + fn else float('nan')

    metrics = {
        'ACC': accuracy_score(y_true, y_pred),
        'ACSA': (benign_recall + mal_recall) / 2,
        'GM': np.sqrt(benign_recall * mal_recall),
        'RECALL': mal_recall,
        'F1-score': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if np.unique(y_true).size == 2:
        precision, recall, _ = precision_recall_curve(y_true, mal_score, pos_label=1)
        metrics['PR AUC'] = auc(recall, precision)
        metrics['ROC AUC'] = roc_auc_score(y_true, mal_score)
    else:
        metrics['PR AUC'] = float('nan')
        metrics['ROC AUC'] = float('nan')
    return metrics, cm


def main():
    parser = argparse.ArgumentParser(description='Evaluate the trained SBCL binary classifier')
    parser.add_argument('--data', required=True, help='dataset root containing val/benign and val/mal')
    parser.add_argument('--checkpoint', required=True, help='second-stage liner_checkpoint.pth.tar')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('--batch-size must be positive and --workers must be nonnegative.')
    if not os.path.isfile(args.checkpoint):
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable.')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = ImageNetLT_val(root=os.path.join(args.data, 'val'), transform=[val_transform])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == 'cuda'))
    model = load_classifier(args.checkpoint, device)

    labels, predictions, scores = [], [], []
    with torch.inference_mode():
        for images, target in loader:
            logits = model(images.to(device, non_blocking=True))
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f'Expected model output of shape [batch, 2], got {tuple(logits.shape)}')
            labels.extend(target.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            scores.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())

    metrics, cm = calculate_metrics(labels, predictions, scores)
    print(f'Samples: {len(labels)} | positive class: mal (1)')
    for name in ('ACC', 'PR AUC', 'ROC AUC', 'ACSA', 'GM', 'RECALL', 'F1-score'):
        print(f'{name}: {metrics[name]:.6f}')
    print('\nConfusion matrix (rows=true, columns=predicted; benign, mal):')
    print(cm)
    print('\nClassification report:')
    print(classification_report(labels, predictions, labels=[0, 1],
                                target_names=CLASS_NAMES, digits=4, zero_division=0))


if __name__ == '__main__':
    main()
