import torch
from torch.utils.data import  DataLoader
from torch.utils.data.distributed import DistributedSampler

from network import Model
from utils.util import RandomDataset,save_checkpoint

import argparse
from tqdm import tqdm
import sys
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")

input_size = 5
output_size = 2
batch_size = 30
data_size = 90
epochs=10

# 2） 配置每个进程的gpu


def train(argv):

    args=parse_args(argv)

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)



    dataset = RandomDataset(input_size, data_size)
    # 3）使用DistributedSampler
    rand_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             sampler=DistributedSampler(dataset),
                             pin_memory=(device == "cuda"))

    model = Model(input_size, output_size)

    # 4) 封装之前要把模型移到对应的gpu
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)

    """
    ------------------------------------------------------------------------------------
    """
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)

    lr_schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion=torch.nn.MSELoss()
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        Model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_schedule.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")

    print("Loading", args.checkpoint,"from epoch",last_epoch)
    print("\n")
    print("The learning rate now is ")


    for epoch in range(0,args.epochs):
        train_lost=0
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        for data in tqdm(rand_loader):
            if torch.cuda.is_available():
                input_var = data
            else:
                input_var = data

            output = model(input_var)
            loss = criterion(output, data)
            optimizer.zero_grad()   #梯度先清零免得叠加
            loss.backward()
            if args.clip_max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm) # 梯度裁剪

            optimizer.step()
        lr_schedule.step(loss)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.modules().state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_schedule.state_dict(),
                },
                is_best,
            )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )

    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--savename", type=str, help="Path to a savename")

    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--lambda",
    #     dest="lmbda",
    #     type=float,
    #     default=1e-2,
    #     help="Bit-rate distortion parameter (default: %(default)s)",
    # )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch_size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--patch-size",
    #     type=int,
    #     nargs=2,
    #     default=(256, 256),
    #     help="Size of the patches to be cropped (default: %(default)s)",
    # )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

if __name__=="__main__":
    train(sys.argv[1:])