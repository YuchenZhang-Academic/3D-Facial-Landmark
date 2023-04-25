import argparse
import torch

class Args(object):
    """
    设置命令行参数的接口
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def set_train_args(self):
        """options for train"""
        self.parser.add_argument("--batch_size", type=int, default=10)
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--weight_decay", type=float, default=1e-4)
        self.parser.add_argument("--epoch", type=int, default=400, help="number of end epoch")
        self.parser.add_argument("--start_epoch", type=int, default=0, help="number of start epoch")
        self.parser.add_argument("--use_GPU", action="store_true", help="identify whether to use gpu")
        self.parser.add_argument("--GPU_id", type=int, default=0, help="device id")
        self.parser.add_argument("--dataset_dir", type=str, default=r"/home/zyc/project/Face/Data/")
        self.parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
        self.parser.add_argument("--print_freq", type=int, default=11,
                            help="print training information frequency (per n iteration)")
        self.parser.add_argument("--save_freq", type=int, default=50, help="save model frequency (per n epoch)")
        self.parser.add_argument("--num_workers", type=int, default=4, help="use n threads to read data")
        self.parser.add_argument("--random_seed", type=int, default=0, help="random seed for split dataset")
        self.parser.add_argument("--pretrain", type=str, default=None, help="the path of pretrain parameters")
        self.parser.add_argument("--organ", type=int, default=0, help="choose organ to train")

        self.opts = self.parser.parse_args()

        if torch.cuda.is_available():
            self.opts.use_GPU = True
            self.opts.GPU_id = torch.cuda.current_device()
            print("use GPU %d to train." % (self.opts.GPU_id))
        else:
            print("use CPU to train.")

    def set_test_args(self):
        """options for inference"""
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--organ", type=int, default=0, help="choose organ to test")
        self.parser.add_argument("--use_GPU", action="store_true", help="identify whether to use gpu")
        self.parser.add_argument("--GPU_id", type=int, default=0, help="device id")
        self.parser.add_argument("--dataset_dir", type=str, default=r"/home/zyc/project/Face/Data/")
        self.parser.add_argument("--weight_path", type=str,
                            default=r"./checkpoints/epoch400.pkl",
                            help="load path for model weight")

        self.opts = self.parser.parse_args()
        if torch.cuda.is_available():
            self.opts.use_GPU = True
            self.opts.GPU_id = torch.cuda.current_device()
            print("use GPU %d to train." % (self.opts.GPU_id))
        else:
            print("use CPU to train.")

    def get_opts(self):
        return self.opts
    
if __name__ == '__main__':
    args = Args()
    args.set_train_args()
    args.get_opts()
