import sklearn.metrics as metrics
import torch


def roc_auc_score(logits, labels, mask=None):
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()
    return metrics.roc_auc_score(labels, logits, average='macro')


def average_precision_score(logits, labels, mask=None):
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()
    return metrics.average_precision_score(labels, logits, average='macro')


def macro_f1_score(logits, labels, mask=None):
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()
    logits = logits > 0.5
    return metrics.f1_score(labels, logits, average='macro')


