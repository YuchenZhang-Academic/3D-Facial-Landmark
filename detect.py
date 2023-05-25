import os
import datetime
import time
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import DetectNet
from dataset import FaceDataset
from arguments import Args

class TrainInterface(object):
    def __init__(self, args) -> None:
        self.args = args  # 命令行参数
        self.img = []
        print("Start Training")

    # @staticmethod
    def __train(self, model, train_loader, optimizer, epoch, num_train, args):
        '''
        model: 需要训练的网络
        train_loader: 训练数据集
        optimizer: 优化器
        epoch: 当前epoch
        num_train: 训练集的数量
        args: 命令行参数
        '''
        model.double()
        model.train()
        device = args.GPU_id
        avg_loss = 0.

        log_file = open(os.path.join(args.checkpoints_dir, "train_log.txt"), "a+")
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_file.write(time)
        log_file.write("\nTraining Epoch: %d\n" %epoch)
        for i, (objs, marks) in enumerate(train_loader):
            if args.use_GPU:
                objs = objs.to(device)
                marks = marks.to(device)
            predicts = model(objs)
            loss = model.cal_loss(marks)
            self.img.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = (avg_loss * i + loss.item()) / (i + 1)
            if i % args.print_freq == 0:
                print("Epoch %d/%d | Iter %d/%d | training loss = %.8f, avg_loss = %.8f" %
                      (epoch, args.epoch, i, num_train//args.batch_size, loss.item(), avg_loss))
            log_file.write("Epoch %d/%d | Iter %d/%d | training loss = %.8f, avg_loss = %.8f\n" %
                      (epoch, args.epoch, i, num_train//args.batch_size, loss.item(), avg_loss))
            log_file.flush()
        log_file.close()

    @staticmethod
    def __validate(model, val_loader, args):

        model.eval()
        # log_file = open(os.path.join(args.checkpoints_dir, "val_log.txt"), "a+")
        # log_file.write("\nValidate Epoch: %d\n" % epoch)
        preds = None
        gts = None
        avg_metric = 0.
        with torch.no_grad():
            for i, (objs, marks) in enumerate(val_loader):
                if args.use_GPU:
                    objs = objs.to(args.GPU_id)
                pred = model(objs)
                loss = model.cal_loss(marks)
                print("Evaluation of validation result: %f" %loss.item())
            # log_file.write("Evaluation of validation result: %f" %loss.item())
            # log_file.flush()
        # log_file.close()
    
    @staticmethod
    def __save_model(model, epoch, args):
        model_name = "epoch%d.pkl" % epoch
        save_dir = os.path.join(args.checkpoints_dir, model_name)
        torch.save(model, save_dir)

    def val(self):
        args = self.args
        model = torch.load(args.weight_path)
        val_dataset = FaceDataset(args.dataset_dir, mode='val', train_val_ratio=0.9)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=True)
        self.__validate(model=model, val_loader=val_loader, args=args)



    def main(self):

        args = self.args
        if not os.path.exists(args.checkpoints_dir):
            os.mkdir(args.checkpoints_dir)
        
        random_seed = args.random_seed
        train_dataset = FaceDataset(args.dataset_dir, seed=random_seed, mode="train", train_val_ratio=0.9)
        val_dataset = FaceDataset(args.dataset_dir, seed=random_seed, mode='val', train_val_ratio=0.9)

        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)

        num_train = len(train_dataset)
        num_val = len(val_dataset)
        with open(os.path.join(args.checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.truncate(0)
        if args.pretrain is None:
            model = DetectNet()
        else:
            model = torch.load(args.pretrain)
        if args.use_GPU:
            model.to(args.GPU_id)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for i in range(args.start_epoch, args.epoch + 1):
            t_start = time.time()
            self.__train(model=model, train_loader=train_loader, optimizer=optimizer, epoch=i, num_train=num_train, args=args)
            t_end = time.time()
            print("Training consumes %.2f second" % (t_end - t_start))

            with open(os.path.join(args.checkpoints_dir, "log.txt"), "a+") as log_file:
                log_file.write("Training consumes %.2f second\n" % (t_end - t_start))
            if i % args.save_freq == 0 or i == args.epoch + 1:
                self.__save_model(model, i, args)
        plt.plot(self.img)
        plt.savefig('loss.png')



if __name__ == '__main__':
    '''
    args = Args()
    args.set_train_args()
    train_interface = TrainInterface(args.get_opts())
    train_interface.main()
    '''
    args = Args()
    args.set_test_args()
    val_interface = TrainInterface(args.get_opts())
    val_interface.val()

    