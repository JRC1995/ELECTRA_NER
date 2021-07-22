import argparse
import json



parser = argparse.ArgumentParser()
parser.add_argument("log", help="path to log file")
parser.add_argument("--metric", choices=["loss", "f1"], help="Metric to use to choose best results", default="f1")
args = parser.parse_args()


# print('Loading pre-trained weights for the model...')
log = json.load(open(args.log))



train_F1s = log['train_F1s']
train_losses = log['train_losses']
val_losses = log['val_losses']
val_F1s = log['val_F1s']
test_losses = log['test_losses']
test_F1s = log['test_F1s']


print("Best train loss (overall): {0}".format(min(train_losses)))
print("Best train F1 (overall): {0}".format(max(train_F1s)))

print("Best test loss (overall): {0}".format(min(test_losses)))
print("Best test F1 (overall): {0}".format(max(test_F1s)))

print("Val loss (best): {0}".format(val_losses[-1]))
print("Val F1 (best): {0}".format(val_F1s[-1]))

if args.metric == "loss":
    idx = val_losses.index(min(val_losses))
else:
    idx = val_F1s.index(max(val_F1s))


print("Train loss (best): {0}".format(train_losses[idx]))
print("Train F1 (best): {0}".format(train_F1s[idx]))

print("Test loss (best): {0}".format(test_losses[idx]))
print("Test F1 (best): {0}".format(test_F1s[idx]))
