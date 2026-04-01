import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.functional import f1_score as f1_score_torch, auroc, accuracy


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)

def calculate_accuracy(outputs, targets, outputs_uncertainty=None, logger=None):
    if outputs_uncertainty is not None:
        # corr between err and uncert
        err2 = F.cross_entropy(outputs, targets, reduction='none').unsqueeze(-1)
        cov = (err2 - err2.mean(axis=0, keepdims=True)) * (outputs_uncertainty - outputs_uncertainty.mean(axis=0, keepdims=True))
        pearson_corr = cov.mean(axis=0) / (err2.std(axis=0, unbiased=False) * outputs_uncertainty.std(axis=0, unbiased=False) + 1e-5)
        pearson_corr = pearson_corr.mean()

    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()
        acc = n_correct_elems / batch_size
        
        f1_macro = f1_score_torch(pred.squeeze(0), targets.long(), average='macro', num_classes=101, multiclass=True)

        if logger is not None:
            logger.add_log("accuracy", acc)
            logger.add_log("f1_macro", f1_macro)
            logger.write_log(0)

        if outputs_uncertainty is not None:
            return {'accuracy': torch.tensor(acc), 'f1_macro': f1_macro, 'uncert_corr': pearson_corr}
        else:
            return {'accuracy': torch.tensor(acc), 'f1_macro': f1_macro}

def calculate_auroc(outputs, targets, logger=None):
    with torch.no_grad():
        all_logits = torch.softmax(outputs, dim=-1)
        auroc_score = auroc(all_logits, targets.long(), average='weighted', num_classes=2)
        acc = accuracy(all_logits, targets.long(), average='weighted', num_classes=2)
        f1_macro = f1_score_torch(all_logits, targets.long(), average='macro', num_classes=2)

        if logger is not None:
            logger.add_log("auroc", auroc_score)
            logger.add_log("accuracy", acc)
            logger.add_log("f1_macro", f1_macro)
            logger.write_log(0)

        return {'auroc': auroc_score, 'accuracy': acc, 'f1_macro': f1_macro}

def calculate_f1(outputs, targets, logger=None):
    all_logits = torch.sigmoid(outputs)
    
    f1_micro = f1_score_torch(all_logits, targets, average='micro', num_labels=23, task="multilabel")
    f1_macro = f1_score_torch(all_logits, targets, average='macro', num_labels=23, task="multilabel")
    f1_samples = f1_score_torch(all_logits, targets, average='none', num_labels=23, task="multilabel")
    f1_weighted = f1_score_torch(all_logits, targets, average='weighted', num_labels=23, task="multilabel")

    if logger is not None:
        logger.add_log("f1_macro", f1_macro)
        logger.add_log("f1_micro", f1_micro)
        logger.add_log("f1_weighted", f1_weighted)
        logger.add_log("f1_samples", f1_samples)
        logger.write_log(0)
    
    return {'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'f1_samples': f1_samples}

def compute_mi(error, uncertainty):
        """Computes mutual information between error and uncertainty.
        Args:
            error: numpy binary array indicating error.
            uncertainty: numpy float array indicating uncertainty.
        Returns:
            mutual_information
        """
        hist_2d, x_edges, y_edges = np.histogram2d(error.ravel(), uncertainty.ravel())
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)  # marginal for x over y
        py = np.sum(pxy, axis=0)  # marginal for y over x
        px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def eval_book(results, truths, test_uncert, logger=None):
    # corr between err and uncert
    err2 = F.cross_entropy(results, truths, reduction='none')
    test_uncert = test_uncert.squeeze()
    cov = (err2 - err2.mean(axis=0, keepdims=True)) * (test_uncert - test_uncert.mean(axis=0, keepdims=True))
    pearson_corr = cov.mean(axis=0) / (err2.std(axis=0) * test_uncert.std(axis=0) + 1e-5)
    pearson_corr = pearson_corr.mean().item()

    if len(results.shape) == 2:
        results = results.max(1)[1]
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    f_score = f1_score(test_preds, test_truth, average='weighted')
    acc = accuracy_score(test_truth, test_preds)
    # Log results
    if logger is not None:
        logger.add_log("f1_score", f_score)
        logger.add_log("accuracy", acc)
        logger.add_log("uncert_corr", pearson_corr)
        logger.write_log(0)
    
    return {'f1_score': torch.tensor(f_score), 
            'accuracy': torch.tensor(acc),
            'uncert_corr': torch.tensor(pearson_corr)}

def eval_mosei(results, truths, logger=None, exclude_zero=False):

    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    # test_preds = (test_preds - np.min(test_preds)) / (np.max(test_preds) - np.min(test_preds)) * 6 - 3
    # test_truth = (test_truth - np.min(test_truth)) / (np.max(test_truth) - np.min(test_truth)) * 6 - 3

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    if torch.var(truths.view(-1).float()) == 0 or torch.var(truths.view(-1).float()) == 0:
        corr = -100.
    else:
        corr = np.corrcoef(test_preds, test_truth)[0][1]
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc = accuracy_score(binary_truth, binary_preds)

    # Log results
    if logger is not None:
        logger.add_log("mae", mae)
        logger.add_log("correlation", corr)
        logger.add_log("f1_score", f_score)
        logger.add_log("accuracy", acc)
        logger.write_log(0)

    # return dict of results
    return {'mae': torch.tensor(mae), 
            'correlation': torch.tensor(corr), 
            'f1_score': torch.tensor(f_score), 
            'accuracy': torch.tensor(acc)}


def eval_mosi(results, truths, logger=None, exclude_zero=False):
    return eval_mosei(results, truths, logger, exclude_zero)


def eval_iemocap(results, truths, logger, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()

        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
            test_truth_i = test_truth[:, emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()

        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds, axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)



