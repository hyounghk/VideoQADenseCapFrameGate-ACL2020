import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

import sys
import time
import json


from qanet.tvqanet import TVQANet
from tvqa_dataset import TVQADataset, pad_collate, prepare_inputs
from config import BaseOptions

import logging
logging.basicConfig()


def mask_logits(target, mask):
    return target * mask 



def IOFSM(selection_greedy, targets, ts_target, ts_target_mask):

    bsz = targets.size(0)
    img_len = selection_greedy.size(1)
    selection_greedy = selection_greedy.view(bsz, 5, -1)
    selection_greedy = selection_greedy[torch.arange(bsz, dtype=torch.long), targets] #(N, Li)



    label = torch.zeros(bsz, img_len).cuda()

    st_list = ts_target["st"].tolist()
    ed_list = ts_target["ed"].tolist()
    for idx, (st, ed) in enumerate(zip(st_list, ed_list)):
        label[idx, st:ed+1] = 1

    label_inv = (label != 1).float()

    rewards_greedy_inv = (selection_greedy * label_inv * ts_target_mask).sum(-1) / (label_inv * ts_target_mask).sum(-1) 

    loss = 1 + rewards_greedy_inv - ((selection_greedy * label).sum(-1) / label.sum(-1))

    return loss.sum(), rewards_greedy_inv.sum(), ((selection_greedy * label).sum(-1) / label.sum(-1)).sum()



def binaryCrossEntropy(max_statement_sm_sigmoid, targets, ts_target, ts_target_mask):

    bsz = targets.size(0)
    max_statement_sm_sigmoid = max_statement_sm_sigmoid.view(bsz, 5, -1)
    img_len = max_statement_sm_sigmoid.size(2)
    max_statement_sm_sigmoid = max_statement_sm_sigmoid[torch.arange(bsz, dtype=torch.long), targets] 

    label = torch.zeros(bsz, img_len).cuda()

    st_list = ts_target["st"].tolist()
    ed_list = ts_target["ed"].tolist()
    for idx, (st, ed) in enumerate(zip(st_list, ed_list)):
        label[idx, st:ed+1] = 1

    loss = nn.functional.binary_cross_entropy_with_logits(max_statement_sm_sigmoid, label, reduction="none")
    loss = mask_logits(loss, ts_target_mask).sum()

    loss *= 0.1

    return loss

def balanced_binaryCrossEntropy(max_statement_sm_sigmoid, targets, ts_target, ts_target_mask):

    bsz = targets.size(0)
    max_statement_sm_sigmoid = max_statement_sm_sigmoid.view(bsz, 5, -1)
    img_len = max_statement_sm_sigmoid.size(2)
    max_statement_sm_sigmoid = max_statement_sm_sigmoid[torch.arange(bsz, dtype=torch.long), targets] #(N, Li)
    label = torch.zeros(bsz, img_len).cuda()

    st_list = ts_target["st"].tolist()
    ed_list = ts_target["ed"].tolist()
    for idx, (st, ed) in enumerate(zip(st_list, ed_list)):
        label[idx, st:ed+1] = 1


    label_inv = (label != 1).float()

    loss = nn.functional.binary_cross_entropy_with_logits(max_statement_sm_sigmoid, label, reduction="none")
    loss_p = mask_logits(loss, label).sum(-1) / label.sum(-1)
    loss_n = mask_logits(loss, label_inv * ts_target_mask).sum(-1) / (label_inv * ts_target_mask).sum(-1) 
    loss = loss_p + loss_n

    return loss.sum()


def train(opt, dset, model, criterion, optimizer, epoch, previous_best_acc):
    dset.set_mode("train")
    model.train()
    train_loader = DataLoader(dset, batch_size=opt.bsz, shuffle=True,
                              collate_fn=pad_collate, num_workers=opt.num_workers, pin_memory=True)

    train_loss = []
    train_loss_iofsm = []
    train_loss_accu = []
    train_loss_ts = []
    train_loss_cls = []
    valid_acc_log = ["batch_idx\tacc\tacc1\tacc2"]
    train_corrects = []
    torch.set_grad_enabled(True)
    max_len_dict = dict(
        max_sub_l=opt.max_sub_l,
        max_vid_l=opt.max_vid_l,
        max_vcpt_l=opt.max_vcpt_l,
        max_qa_l=opt.max_qa_l,
        max_dc_l=opt.max_dc_l,
    )


    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader)):
        timer_start = time.time()
        model_inputs, targets, qids = prepare_inputs(batch, max_len_dict=max_len_dict, device=opt.device)
        try:
            timer_start = time.time()
            outputs, max_statement_sm_sigmoid_ = model(model_inputs)
            
            max_statement_sm_sigmoid, max_statement_sm_sigmoid_selection = max_statement_sm_sigmoid_

            temporal_loss = balanced_binaryCrossEntropy(max_statement_sm_sigmoid, targets, model_inputs["ts_label"], model_inputs["ts_label_mask"])


            cls_loss = criterion(outputs, targets)

            iofsm_loss, _, _ = IOFSM(max_statement_sm_sigmoid_selection, targets, model_inputs["ts_label"], model_inputs["ts_label_mask"])

            att_loss_accu = 0

            loss = cls_loss + temporal_loss + iofsm_loss

            timer_start = time.time()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.data.item())
            train_loss_iofsm.append(float(iofsm_loss))
            train_loss_ts.append(float(temporal_loss))

            train_loss_cls.append(cls_loss.item())
            pred_ids = outputs.data.max(1)[1]
            train_corrects += pred_ids.eq(targets.data).tolist()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: ran out of memory, skipping batch")
            else:
                print("RuntimeError {}".format(e))
                sys.exit(1)

        if batch_idx % opt.log_freq == 0:
            niter = epoch * len(train_loader) + batch_idx
            if batch_idx == 0:  
                train_acc = 0
                train_loss = 0
                train_loss_iofsm = 0
                train_loss_ts = 0
                train_loss_cls = 0
            else:
                train_acc = sum(train_corrects) / float(len(train_corrects))
                train_loss = sum(train_loss) / float(len(train_corrects))
                train_loss_iofsm = sum(train_loss_iofsm) / float(len(train_corrects))
                train_loss_cls = sum(train_loss_cls) / float(len(train_corrects))
                train_loss_ts = sum(train_loss_ts) / float(len(train_corrects))


            valid_acc, valid_loss, qid_corrects, valid_acc1, valid_acc2, submit_json_val = \
                validate(opt, dset, model, criterion, mode="valid")

            valid_log_str = "%02d\t%.4f\t%.4f\t%.4f" % (batch_idx, valid_acc, valid_acc1, valid_acc2)
            valid_acc_log.append(valid_log_str)

            if valid_acc > previous_best_acc:
                with open("best_github.json", 'w') as cqf:
                    json.dump(submit_json_val, cqf)
                previous_best_acc = valid_acc
                if epoch >= 10:
                    torch.save(model.state_dict(), os.path.join("./results/best_valid_to_keep", "best_github_7420.pth"))

            print("Epoch {:02d} [Train] acc {:.4f} loss {:.4f} loss_iofsm {:.4f} loss_ts {:.4f} loss_cls {:.4f}"
                  "[Val] acc {:.4f} loss {:.4f}"
                  .format(epoch, train_acc, train_loss, train_loss_iofsm, train_loss_ts, train_loss_cls,
                          valid_acc, valid_loss))


            torch.set_grad_enabled(True)
            model.train()
            dset.set_mode("train")
            train_corrects = []
            train_loss = []
            train_loss_iofsm = []
            train_loss_ts = []
            train_loss_cls = []

        timer_dataloading = time.time()


    with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
        f.write("\n".join(valid_acc_log) + "\n")

    return previous_best_acc


def validate(opt, dset, model, criterion, mode="valid"):
    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False,
                              collate_fn=pad_collate, num_workers=opt.num_workers, pin_memory=True)

    submit_json_val = {}
    valid_qids = []
    valid_loss = []
    valid_corrects = []
    max_len_dict = dict(
        max_sub_l=opt.max_sub_l,
        max_vid_l=opt.max_vid_l,
        max_vcpt_l=opt.max_vcpt_l,
        max_qa_l=opt.max_qa_l,
        max_dc_l=opt.max_dc_l,
    )
    for val_idx, batch in enumerate(valid_loader):
        model_inputs, targets, qids = prepare_inputs(batch, max_len_dict=max_len_dict, device=opt.device)
        outputs, _= model(model_inputs)

        loss = criterion(outputs, targets)

        valid_qids += [int(x) for x in qids]
        valid_loss.append(loss.data.item())
        pred_ids = outputs.data.max(1)[1]

        for qdix, q_id in enumerate(model_inputs['qid']):
            q_id_str = str(q_id)
            submit_json_val[q_id_str] = int(pred_ids[qdix].item())

        valid_corrects += pred_ids.eq(targets.data).tolist()

    acc_1st, acc_2nd = 0., 0. 
    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    valid_loss = sum(valid_loss) / float(len(valid_corrects))
    qid_corrects = ["%d\t%d" % (a, b) for a, b in zip(valid_qids, valid_corrects)]
    return valid_acc, valid_loss, qid_corrects, acc_1st, acc_2nd, submit_json_val


def main():
    opt = BaseOptions().parse()
    torch.manual_seed(opt.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(opt.seed)

    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)

    model = TVQANet(opt)


    if opt.device.type == "cuda":
        print("CUDA enabled.")
        if len(opt.device_ids) > 1:
            print("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids, output_device=0)  # use multi GPU
        model.to(opt.device)


    # model.load_state_dict(torch.load("./path/best_release_7420.pth"))


    criterion = nn.CrossEntropyLoss(reduction="sum").to(opt.device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr,
        weight_decay=opt.wd)

    best_acc = 0.
    start_epoch = 0
    early_stopping_cnt = 0
    early_stopping_flag = False

    for epoch in range(start_epoch, opt.n_epoch):
        if not early_stopping_flag:

            niter = epoch * np.ceil(len(dset) / float(opt.bsz))

            cur_acc = train(opt, dset, model, criterion, optimizer, epoch, best_acc)

            is_best = cur_acc > best_acc
            best_acc = max(cur_acc, best_acc)
            if not is_best:
                early_stopping_cnt += 1
                if early_stopping_cnt >= opt.max_es_cnt:
                    early_stopping_flag = True
            else:
                early_stopping_cnt = 0
        else:
            print("=> early stop with valid acc %.4f" % best_acc)
            break  

        if epoch == 10:
            for g in optimizer.param_groups:
                g['lr'] = 0.0002

    return opt.results_dir.split("/")[1]


if __name__ == "__main__":
    results_dir = main()

