import os, collections
import numpy as np
import sys

def do(modelName):
    path = "/home/jmanotum/workspace/projects/trippy-dst/"
    files = [f for f in os.listdir(path) if modelName in f]
    # print(files)
    memo = collections.defaultdict(list)
    for file in files:
        try:
            fold = "".join(file.split("_")[:-1])
            results = open(path+file+"/eval_pred_test.log").read().splitlines()[-6:]
            acc = max([float(res.split(" ")[3][:-1]) * 100 for res in results])
            # for res in results:
            #     if "final" in res:
            #         acc =
            memo[fold].append(acc)
        except:
            continue
    for fold in sorted(memo):
        print(fold+","+"%.2f (%.1f)" % (np.mean(np.array(memo[fold])), np.std(np.array(memo[fold]))))


do(sys.argv[1])
