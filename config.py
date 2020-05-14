import os
import time
import torch
import argparse

import _init_paths
from utils.utils import mkdirp, load_json, save_json_pretty, make_zipfile


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        self.parser.add_argument("--results_dir_base", type=str, default="results/results")
        self.parser.add_argument("--log_freq", type=int, default=4000, help="print, save training info")
        self.parser.add_argument("--seed", type=int, default=2018, help="random seed")

        # training config
        self.parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        self.parser.add_argument("--wd", type=float, default=3e-7, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=5, help="number of epochs to early stop")
        self.parser.add_argument("--bsz", type=int, default=16, help="mini-batch size")
        self.parser.add_argument("--test_bsz", type=int, default=16, help="mini-batch size for testing")
        self.parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU ids to run the job")
        self.parser.add_argument("--num_workers", type=int, default=1,
                                 help="num subprocesses used to load the data, 0: use main process")

        self.parser.add_argument("--input_streams", type=str, nargs="+",
                                 choices=["sub", "vfeat", "dense"],
                                 help="input streams for the model, will use both `vcpt` and `sub` streams")

        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--use_core_driver_text", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")

        self.parser.add_argument("--learn_word_embedding", action="store_true", help="fix word embedding")
        self.parser.add_argument("--clip", type=float, default=10., help="perform gradient clip")
        self.parser.add_argument("--resume", type=str, default="", help="path to latest checkpoint")

        self.parser.add_argument("--scale", type=float, default=10.,
                                 help="multiplier to be applied to similarity score")
        
        
        self.parser.add_argument("--max_sub_l", type=int, default=50,
                                 help="maxmimum length of all sub sentence 97.71 under 50 for 3 sentences")
        self.parser.add_argument("--max_dc_l", type=int, default=100,
                                 help="maxmimum length of all sub sentence 97.71 under 50 for 3 sentences")
        self.parser.add_argument("--max_para_l", type=int, default=70,
                                 help="maxmimum length of all sub sentence 97.71 under 50 for 3 sentences")
        self.parser.add_argument("--max_vid_l", type=int, default=300,
                                 help="maxmimum length of all video sequence")
        self.parser.add_argument("--max_vcpt_l", type=int, default=300,
                                 help="maxmimum length of video seq, 94.25% under 20")
        self.parser.add_argument("--max_q_l", type=int, default=20,
                                 help="maxmimum length of question, 93.91% under 20")  # 25
        self.parser.add_argument("--max_a_l", type=int, default=15,
                                 help="maxmimum length of answer, 98.20% under 15")
        self.parser.add_argument("--max_qa_l", type=int, default=40,
                                 help="maxmimum length of answer, 99.7% <= 40")

        self.parser.add_argument("--embedding_size", type=int, default=300, help="word embedding dim")
        self.parser.add_argument("--hsz", type=int, default=128, help="hidden size.")
        self.parser.add_argument("--vocab_size", type=int, default=0, help="vocabulary size")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

        self.parser.add_argument("--input_encoder_n_blocks", type=int, default=1)
        self.parser.add_argument("--input_encoder_n_conv", type=int, default=2)
        self.parser.add_argument("--input_encoder_kernel_size", type=int, default=7)
        self.parser.add_argument("--input_encoder_n_heads", type=int, default=0,
                                 help="number of self-attention heads, 0: do not use it")

        self.parser.add_argument("--cls_encoder_n_blocks", type=int, default=1)
        self.parser.add_argument("--cls_encoder_n_conv", type=int, default=2)
        self.parser.add_argument("--cls_encoder_kernel_size", type=int, default=5)
        self.parser.add_argument("--cls_encoder_n_heads", type=int, default=0,
                                 help="number of self-attention heads, 0: do not use it")

    def add_path_cfg(self):
        opt = self.opt

        base_path = "../../TVQA/TVQA/data"
        opt.train_path = os.path.join(base_path, "tvqa_train_processed.json")
        opt.valid_path = os.path.join(base_path, "tvqa_val_processed.json")
        opt.test_path = os.path.join(base_path, "tvqa_test_public_processed.json")
        

        base_path = "" 
        opt.glove_path = "../../TVQA/TVQA/data/glove.6B.300d.txt"
        opt.word2idx_path = "cache/word2idx.pickle"
        opt.idx2word_path = "cache/idx2word.pickle"
        opt.vocab_embedding_path = "cache/vocab_embedding.pickle"

        vfeat_flag = "vfeat" in opt.input_streams
        if vfeat_flag:
            opt.vfeat_size = 300
        else:
            opt.vfeat_size = None

        self.opt = opt


    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        if opt.input_streams is None:
            raise ValueError("input_streams must be set")

        if opt.resume:  
            if os.path.isfile(opt.resume):
                opt.results_dir = os.path.dirname(opt.resume)
            else:
                raise ValueError("Wrong argument to --resume, %s" % opt.resume)
        else:
            opt.results_dir = opt.results_dir_base + time.strftime("_%Y_%m_%d_%H_%M_%S")

        self.opt = opt
        mkdirp(opt.results_dir)
        code_dir = os.path.dirname(os.path.realpath(__file__))
        code_zip_filename = os.path.join(opt.results_dir, "code.zip")

        self.add_path_cfg()

        opt.device = torch.device("cuda:%d" % opt.device_ids[0] if opt.device >= 0 else "cpu")
        opt.h5driver = None if opt.no_core_driver else "core"
        opt.vfeat_flag = "vfeat" in opt.input_streams
        opt.sub_flag = "sub" in opt.input_streams
        opt.dense_flag = "dense" in opt.input_streams
        self.opt = opt
        return opt

