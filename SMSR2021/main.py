import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    # manual_Seed 在需要生成随机数据的实验中，每次实验都需要生成数据。
    # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    # 创建相关文件夹，并写好log日志
    if checkpoint.ok:
        loader = data.Data(args)                ## data loader
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()

        checkpoint.done()
