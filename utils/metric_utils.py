import numpy as np


def eval_stats(batch_predictions, batch_labels,  batch_mask, idx2labels):

    tags = idx2labels

    tp = 0
    pred_len = 0
    gold_len = 0

    for i in range(0, len(batch_labels)):
        j = 0
        true_seq_len = sum(batch_mask[i])
        while j < true_seq_len:

            init_j = j

            if tags[batch_labels[i][j]][0] == 'B':
                gold_len += 1
                chunk_label = []
                chunk_prediction = []
                chunk_label.append(batch_labels[i][j])
                chunk_prediction.append(batch_predictions[i][j])

                j += 1

                while j < true_seq_len:
                    if tags[batch_labels[i][j]][0] in ['I']:
                        chunk_label.append(batch_labels[i][j])
                        chunk_prediction.append(batch_predictions[i][j])
                        j += 1
                    else:
                        break

                label_entity = np.asarray(chunk_label, dtype=np.int32)
                prediction_entity = np.asarray(chunk_prediction, dtype=np.int32)

                if np.all(np.equal(label_entity, prediction_entity)):
                    tp += 1

            elif tags[batch_labels[i][j]][0] == 'O':
                j += 1

        j = 0

        while j < true_seq_len:

            if tags[batch_predictions[i][j]][0] in ['B']:
                pred_len += 1
            j += 1

    return tp, pred_len, gold_len


def compute_F1(tp, pred_len, gold_len):

    prec = tp/pred_len if pred_len > 0 else 0
    rec = tp/gold_len if gold_len > 0 else 0
    F1 = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0

    return prec, rec, F1
