from importlib import import_module
from dataloader import MSDataLoader
import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

class Data:
    def __init__(self, args):
        self.loader_train = None
        #如果用于训练，则导入数据
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())     ## load the right dataset loader module
            trainset = getattr(module_train, args.data_train)(args)             ## load the dataset, args.data_train is the  dataset name
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        # 如果用于测试，则选取一个数据集
        if args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100','videos_img']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test,train=False)
            # getattr(a, 'b')的作用就和a.b是一样的
        else:
            module_test = import_module('data.' + args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )



if __name__ == '__main__':
    torch.manual_seed(args.seed)
    # manual_Seed 在需要生成随机数据的实验中，每次实验都需要生成数据。
    # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    # 创建相关文件夹，并写好log日志
    if checkpoint.ok:
        loader = data.Data(args)  