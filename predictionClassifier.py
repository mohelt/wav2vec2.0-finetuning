# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from fairseqClassifier import UrbanPredictor, MyDataset
import numpy as np
import scipy.stats
import torchaudio

def systemID(uttID):
    return uttID.split('-')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model.')
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--outfile', type=str, required=False, default='answer.txt', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    args = parser.parse_args()
    
    cp_path = args.fairseq_base_model
    my_checkpoint = args.finetuned_checkpoint
    datadir = args.datadir
    outfile = args.outfile

    system_csv_path = os.path.join(datadir, 'sets/')

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = UrbanPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint,map_location="cpu"))

    wavdir = os.path.join(datadir, 'URBAN')
    validlist = os.path.join(datadir, 'sets/URBAN/testurban.txt')

    print('Loading data')
    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=1, collate_fn=validset.collate_fn)

    total_loss = 0.0
    num_steps = 0.0
    predictions = { }  # filename : prediction
    criterion = nn.BCELoss()
    print('Starting prediction')


    threshold = 0.5  # set threshold value

    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames = data        
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        output = outputs.cpu().detach().numpy()[0]
        output = 1 if output > threshold else 0  # map to 1 or 0 based on threshold
        predictions[filenames[0]] = output  ## batch size = 1

    true_MOS = { }
    validf = open(validlist, 'r')
    for line in validf:
        parts = line.strip().split(',')
        uttID = parts[0]
        MOS = float(parts[1])
        true_MOS[uttID] = MOS

    ## compute correls.
    sorted_uttIDs = sorted(predictions.keys())
    ts = []
    ps = []
    for uttID in sorted_uttIDs:
        t = true_MOS[uttID]
        p = predictions[uttID]
        ts.append(t)
        ps.append(p)

    truths = np.array(ts)
    preds = np.array(ps)

    ### UTTERANCE
    MSE=np.mean((truths-preds)**2)
    print('[UTTERANCE] Test error= %f' % MSE)
    LCC=np.corrcoef(truths, preds)
    print('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(truths.T, preds.T)
    print('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(truths, preds)
    print('[UTTERANCE] Kendall Tau rank correlation coefficient= %f' % KTAU[0])
    ## generate answer.txt for codalab
    ans = open(outfile, 'w')
    for k, v in predictions.items():
        outl = k.split('.')[0] + ',' + str(v) + '\n'
        ans.write(outl)
    ans.close()
if __name__ == '__main__':
    main()
