#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import math
import time
import json
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from build_dataset import build_dataset, add_args_to_parser, get_dataset_path
from lib import wrn, transform, utils
from config import config


class TrainArgParser(utils.BaseParser):
    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--alg",
            "-a",
            default="VAT",
            type=str,
            help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
        parser.add_argument(
            "--em",
            default=0,
            type=float,
            help=
            "coefficient of entropy minimization. If you try VAT + EM, set 0.06"
        )
        parser.add_argument('--name', default="run")
        parser.add_argument("--validation",
                            default=25000,
                            type=int,
                            help="validate at this interval (default 25000)")
        parser.add_argument("--output",
                            "-o",
                            default="./exp_res",
                            type=str,
                            help="output dir")
        add_args_to_parser(parser)

        return parser


def main():
    args = TrainArgParser()

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    condition = {}
    exp_name = ""

    condition["dataset"] = args.dataset
    exp_name += str(args.dataset)

    condition["algorithm"] = args.alg
    exp_name += "_" + str(args.alg)

    if args.em > 0:
        exp_name += "_em"
    condition["entropy_maximization"] = args.em

    base_save_path = os.path.join("train_output", exp_name)
    run_num = 0
    get_save_path = lambda: os.path.join(base_save_path, args.name + "_{}".
                                         format(run_num))
    while os.path.exists(get_save_path()):
        run_num += 1

    save_path = get_save_path()

    writer = SummaryWriter(log_dir=os.path.join(save_path, "tb"))
    writer.add_text('args', args.as_markdown(), 0)

    logger = utils.get_logger(os.path.join(save_path, "train.log"))
    args.print_params(logger.info)

    build_dataset(args.dataset, nlabels=args.nlabels)

    dataset_cfg = config[args.dataset]
    transform_fn = transform.transform(
        *dataset_cfg["transform"])  # transform function (flip, crop, noise)

    dataset_path = get_dataset_path(args.dataset, args.seed, args.nlabels)

    l_train_dataset = dataset_cfg["dataset"](dataset_path, "l_train")
    u_train_dataset = dataset_cfg["dataset"](dataset_path, "u_train")
    val_dataset = dataset_cfg["dataset"](dataset_path, "val")
    test_dataset = dataset_cfg["dataset"](dataset_path, "test")

    logger.info(
        "labeled data : {}, unlabeled data : {}, training data : {}".format(
            len(l_train_dataset), len(u_train_dataset),
            len(l_train_dataset) + len(u_train_dataset)))
    logger.info("validation data : {}, test data : {}".format(
        len(val_dataset), len(test_dataset)))
    condition["number_of_data"] = {
        "labeled": len(l_train_dataset),
        "unlabeled": len(u_train_dataset),
        "validation": len(val_dataset),
        "test": len(test_dataset)
    }

    class RandomSampler(torch.utils.data.Sampler):
        """ sampling without replacement """

        # pylint: disable=super-init-not-called
        def __init__(self, num_data, num_sample):
            iterations = num_sample // num_data + 1
            self.indices = torch.cat([
                torch.randperm(num_data) for _ in range(iterations)
            ]).tolist()[:num_sample]

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    shared_cfg = config["shared"]

    batch_size = shared_cfg["batch_size"]
    if args.alg != "supervised":
        # batch size = 0.5 x batch size
        batch_size //= 2

    l_loader = DataLoader(l_train_dataset,
                          batch_size,
                          drop_last=True,
                          sampler=RandomSampler(
                              len(l_train_dataset),
                              shared_cfg["iteration"] * batch_size))

    u_loader = DataLoader(u_train_dataset,
                          shared_cfg["batch_size"] // 2,
                          drop_last=True,
                          sampler=RandomSampler(
                              len(u_train_dataset), shared_cfg["iteration"] *
                              shared_cfg["batch_size"] // 2))

    val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)

    logger.info("maximum iteration : {}".format(
        min(len(l_loader), len(u_loader))))

    alg_cfg = config[args.alg]
    logger.info("parameters : {}".format(alg_cfg))
    condition["h_parameters"] = alg_cfg

    model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])

    trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
    logger.info("trainable parameters : {}".format(trainable_paramters))

    # pylint: disable=ungrouped-imports
    if args.alg == "VAT":  # virtual adversarial training
        from lib.algs.vat import VAT
        ssl_obj = VAT(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
    elif args.alg == "PL":  # pseudo label
        from lib.algs.pseudo_label import PL
        ssl_obj = PL(alg_cfg["threashold"])
    elif args.alg == "MT":  # mean teacher
        from lib.algs.mean_teacher import MT
        t_model = wrn.WRN(2, dataset_cfg["num_classes"],
                          transform_fn).to(device)
        t_model.load_state_dict(model.state_dict())
        ssl_obj = MT(t_model, alg_cfg["ema_factor"])
    elif args.alg == "PI":  # PI Model
        from lib.algs.pimodel import PiModel
        ssl_obj = PiModel()
    elif args.alg == "ICT":  # interpolation consistency training
        from lib.algs.ict import ICT
        t_model = wrn.WRN(2, dataset_cfg["num_classes"],
                          transform_fn).to(device)
        t_model.load_state_dict(model.state_dict())
        ssl_obj = ICT(alg_cfg["alpha"], t_model, alg_cfg["ema_factor"])
    elif args.alg == "MM":  # MixMatch
        from lib.algs.mixmatch import MixMatch
        ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
    elif args.alg == "supervised":
        pass
    else:
        raise ValueError("{} is unknown algorithm".format(args.alg))

    logger.info("### start training ###")
    iteration = 0
    maximum_val_acc = 0
    train_cls_loss = utils.AverageMeter()
    train_ssl_loss = utils.AverageMeter()
    s = time.time()
    for l_data, u_data in zip(l_loader, u_loader):
        iteration += 1
        l_input, target = l_data
        l_input, target = l_input.to(device).float(), target.to(device).long()

        if args.alg != "supervised":  # for ssl algorithm
            u_input, dummy_target = u_data
            u_input, dummy_target = u_input.to(
                device).float(), dummy_target.to(device).long()

            target = torch.cat([target, dummy_target], 0)
            unlabeled_mask = (target == -1).float()

            inputs = torch.cat([l_input, u_input], 0)
            outputs = model(inputs)

            # ramp up exp(-5(1 - t)^2)
            coef = alg_cfg["consis_coef"] * math.exp(
                -5 * (1 - min(iteration / shared_cfg["warmup"], 1))**2)
            ssl_loss = ssl_obj(inputs, outputs.detach(), model,
                               unlabeled_mask) * coef

        else:
            outputs = model(l_input)
            coef = 0
            ssl_loss = torch.zeros(1).to(device)

        # supervised loss
        cls_loss = F.cross_entropy(outputs,
                                   target,
                                   reduction="none",
                                   ignore_index=-1).mean()

        loss = cls_loss + ssl_loss

        if args.em > 0:
            loss -= args.em * (
                (outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1) *
                unlabeled_mask).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = target.size(0)
        train_cls_loss.update(cls_loss.item(), n)
        train_ssl_loss.update(ssl_loss.item(), n)

        if args.alg == "MT" or args.alg == "ICT":
            # parameter update with exponential moving average
            ssl_obj.moving_average(model.parameters())
        # display
        if iteration % 100 == 0:
            wasted_time = time.time() - s
            rest = (shared_cfg["iteration"] -
                    iteration) / 100 * wasted_time / 60
            logger.info(
                "iteration [{}/{}] cls loss : {:.6e}, SSL loss : {:.6e}, coef "
                ": {:.5e}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {}".
                format(iteration, shared_cfg["iteration"], train_cls_loss.avg,
                       train_ssl_loss.avg, coef, 100 / wasted_time, rest,
                       optimizer.param_groups[0]["lr"]))
            writer.add_scalar('train/cls_loss', train_cls_loss.avg, iteration)
            writer.add_scalar('train/ssl_loss', train_ssl_loss.avg, iteration)
            writer.add_scalar('train/coef', coef, iteration)
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"],
                              iteration)
            s = time.time()

        # validation
        if (iteration %
                args.validation) == 0 or iteration == shared_cfg["iteration"]:
            with torch.no_grad():
                model.eval()
                logger.info("### validation ###")
                sum_acc = 0.
                s = time.time()
                for j, data in enumerate(val_loader):
                    inp, target = data
                    inp, target = inp.to(device).float(), target.to(
                        device).long()

                    output = model(inp)

                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()
                    if ((j + 1) % 10) == 0:
                        d_p_s = 10 / (time.time() - s)
                        logger.info(
                            "[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec"
                            .format(j + 1, len(val_loader), d_p_s,
                                    (len(val_loader) - j - 1) / d_p_s),
                            "\r",
                            end="")
                        s = time.time()
                acc = sum_acc / float(len(val_dataset))
                logger.info("validation accuracy : {}".format(acc))
                writer.add_scalar('validation/accuracy', acc, iteration)
                # test
                if maximum_val_acc < acc:
                    logger.info("### test ###")
                    maximum_val_acc = acc
                    sum_acc = 0.
                    s = time.time()
                    for j, data in enumerate(test_loader):
                        inp, target = data
                        inp, target = inp.to(device).float(), target.to(
                            device).long()
                        output = model(inp)
                        pred_label = output.max(1)[1]
                        sum_acc += (pred_label == target).float().sum()
                        if ((j + 1) % 10) == 0:
                            d_p_s = 100 / (time.time() - s)
                            logger.info(
                                "[{}/{}] time : {:.1f} data/sec, rest : {:.2f} "
                                "sec".format(j + 1, len(test_loader), d_p_s,
                                             (len(test_loader) - j - 1) /
                                             d_p_s),
                                "\r",
                                end="")
                            s = time.time()
                    test_acc = sum_acc / float(len(test_dataset))
                    logger.info("test accuracy : {}".format(test_acc))
                    writer.add_scalar('test/accuracy', test_acc, iteration)
                    # torch.save(model.state_dict(),
                    #            os.path.join(save_path, "best_model.pth"))
            model.train()
            s = time.time()
        # lr decay
        if iteration == shared_cfg["lr_decay_iter"]:
            optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]

    logger.info("final test acc : {}".format(test_acc))
    condition["test_acc"] = test_acc.item()

    exp_name += str(int(time.time()))  # unique ID
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    with open(os.path.join(args.output, exp_name + ".json"), "w") as f:
        json.dump(condition, f)


if __name__ == '__main__':
    main()
