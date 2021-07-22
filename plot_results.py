import argparse
import json
import matplotlib.pyplot as plt
import matplotlib



parser = argparse.ArgumentParser()
parser.add_argument("--log", nargs='+', help="path to log file(s) to plot file1 file2 ...", required=True)
parser.add_argument("--labels", nargs='+', help="customized labels for logs in the same order label1 label2 ...")
parser.add_argument("-r", action='store_false', default=True, help="exclude training results")
parser.add_argument("-v", action='store_false', default=True, help="exclude validation results")
parser.add_argument("-t", action='store_false', default=True, help="exclude testing results")
parser.add_argument("--metric", choices=["loss", "f1"], help="Metric to use to choose best results", default="f1")
parser.add_argument("--title", help="Title of the plot")
parser.add_argument("--out", help="Output file to save results", default="plot.jpg")
args = parser.parse_args()

logs = []
for log in args.log:
    log = json.loads(open(log, 'r').read())
    logs.append(log)

metrickey_map = {
    'f1': 'F1s',
    'loss': 'losses'
}

cmap = matplotlib.cm.get_cmap('tab10').colors
style = {
    'train': '-^',
    'val': '-o',
    'test': '-s'

}

matplotlib.rcParams.update({'font.size': 10})


for lidx, log in enumerate(logs):
    color = cmap[lidx % 10]
    setkey_list = ['train', 'val', 'test']
    plotit = [args.r, args.v, args.t]
    
    marked = False
    for idx, dset in enumerate(setkey_list):
        if plotit[idx]:
            key = dset + '_' + metrickey_map[args.metric]
            vals = log[key]
            x_seq = list(range(len(vals)))
            best = vals.index(max(vals))
            if args.labels:
                label = None
            else:
                label=dset
            if args.labels and not marked:
                plt.plot([0], vals[0], linewidth=2, label=args.labels[lidx], color=color)
                marked = True
            plt.plot(x_seq, vals, style[dset], label=label, linewidth=2, color=color)
            # plt.text(best - 0.05, vals[best] - 0.03, "{:.3f}".format(vals[best]))
            # plt.plot(best, vals[best], 'o', linewidth=3, color='black')

plt.xlabel("Epoch", fontsize=14)
plt.ylabel(args.metric.capitalize(), fontsize=14)
if args.title:
    title = args.title
else:
    title = args.log[lidx]
plt.title(title)
plt.legend()
plt.tight_layout()
plt.savefig(args.out)

# train_F1s = log['train_F1s']
# train_losses = log['train_losses']
# val_losses = log['val_losses']
# val_F1s = log['val_F1s']
# test_losses = log['test_losses']
# test_F1s = log['test_F1s']