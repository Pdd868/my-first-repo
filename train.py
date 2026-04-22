from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

import warnings
warnings.filterwarnings("ignore")

from terminaltables import AsciiTable

import os
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

"""配置参数：
--model_def
config/yolov3-custom.cfg
--data_config
config/custom.data
--pretrained_weights
weights/darknet53.conv.74
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs") #训练次数
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")   #batch的大小
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")#在每一步（更新模型参数）之前累积梯度的次数”
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file") #模型的配置文件
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file") #数据的配置文件
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model") #预训练文件
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")#数据加载过程中应使用的CPU线程数。
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")#隔多少个epoch保存一次模型权重
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")#多少个epoch进行一次验证集的验证
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")#允许多尺寸特征图融合的训练
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")#日志文件

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)#model.apply(fn)表示将fn函数应用到神经网络的各个模块上，包括该神经网络本身。这通常在初始化神经网络的参数时使用，本处用于初始化神经网络的权值

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"): #用于检查字符串是否以指定的后缀结束。如果字符串以指定的后缀结束，则返回True，否则返回False。
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,  #1个样本打包成一个batch进行加载
        shuffle=True,               #对数据进行随机打乱，
        num_workers=opt.n_cpu,      #用于指定子进程的数量，用于并行地加载数据。默认情况下，num_workers的值为0，表示没有使用子进程，所有数据都会在主进程中加载。当设置num_workers大于0时，DataLoader会创建指定数量的子进程，每个子进程都会负责加载一部分数据，然后主进程负责从这些子进程中获取数据。
                                    # 使用子进程可以加快数据的加载速度，因为每个子进程可以并行地加载一部分数据，从而充分利用多核CPU的计算能力。但是需要注意的是，使用子进程可能会导致数据的顺序被打乱，因此如果需要保持数据的原始顺序，应该将shuffle参数设置为False。
                                    # num_workers的值应该根据具体情况进行调整。如果数据集较大，可以考虑增加num_workers的值以充分利用计算机的资源。但是需要注意的是，如果num_workers的值过大，可能会导致内存消耗过大或者CPU负载过重，从而影响程序的性能。因此，需要根据实际情况进行调整。
        pin_memory=True,            #指定是否将加载进内存的数据的指针固定（pin），这个参数在某些情况下可以提高数据加载的速度。
                                    # 当设置pin_memory=True时，DataLoader会将加载进内存的数据的指针固定，即不进行移动操作。这样做的目的是为了提高数据传输的效率。因为当数据从磁盘或者网络等地方传输到内存中时，如果指针不固定，可能会导致数据在传输过程中被移动，从而需要重新读取，浪费了时间。而固定指针可以避免这种情况的发生，从而提高了数据传输的效率。
                                    # 需要注意的是，pin_memory参数的效果与操作系统和硬件的性能有关。在一些高性能的计算机上，固定指针可能并不会带来太大的性能提升。但是在一些内存带宽较小的计算机上，固定指针可能会显著提高数据加载的效率。因此，需要根据实际情况进行调整。
        collate_fn=dataset.collate_fn,
                                    # collate_fn是一个函数，用于对每个batch的数据进行合并。这个函数的输入是一个batch的数据，输出是一个合并后的数据。
                                    # collate_fn函数的主要作用是对每个batch的数据进行预处理，例如将不同数据类型的张量合并成一个张量，或者对序列数据进行padding操作等。这样可以使得每个batch的数据格式一致，便于模型进行训练。
                                    # 在默认情况下，collate_fn函数会将每个batch的数据按照第一个元素的张量形状进行合并。例如，如果一个batch的数据中第一个元素的张量形状是[
                                    # 3, 224, 224]，那么collate_fn函数会将该batch的所有数据都调整为这个形状。
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))    #Variable类是PyTorch中的一个包装器，它将张量和它们的梯度信息封装在一起。当我们对一个张量进行操作时，PyTorch会自动地创建一个对应的Variable对象，其中包含了原始张量、梯度等信息。通过使用Variable，我们可以方便地进行自动微分和优化。
            targets = Variable(targets.to(device), requires_grad=False)
            print ('imgs',imgs.shape)
            print ('targets',targets.shape)
            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
