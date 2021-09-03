from dataset.data_builder import build_data
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch import autograd
from tqdm import tqdm
import math
import cv2
import torch
import numpy as np
import yaml
import os
import argparse
from nets.encoder import vae_encoder
from nets.decoder import vae_decoder
from loss.losses import vae_loss

log_file = None


def to_log(s, output=True, end="\n"):
    global log_file
    if output:
        print(s, end=end)
    print(s, file=log_file, end=end)


def open_config(root):
    f = open(os.path.join(root, "config.yaml"))
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load(models, epoch, root):
    def _detect_latest():
        checkpoints = os.listdir(os.path.join(root, "logs"))
        checkpoints = [f for f in checkpoints if f.startswith("E_epoch-") and f.endswith(".pth")]
        checkpoints = [int(f[len("E_epoch-"):-len(".pth")]) for f in checkpoints]
        checkpoints = sorted(checkpoints)
        _epoch = checkpoints[-1] if len(checkpoints) > 0 else None
        return _epoch

    if epoch == -1:
        epoch = _detect_latest()
    if epoch is None:
        return -1
    for name, model in models.items():
        ckpt = torch.load(os.path.join(root, "logs/" + name + "_epoch-{}.pth".format(epoch)))
        ckpt = {k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        to_log("load model: {} from epoch: {}".format(name, epoch))
    # print("loaded from epoch: {}".format(epoch))
    return epoch


# BCHW
def batch_image_merge(image):
    # image = torch.cat(image.split(4, 0), 2)
    image = torch.cat(image.split(1, 0), 3)
    # CH'W'
    return image


def imagetensor2np(x):
    x = torch.round((x + 1) / 2 * 255).clamp(0, 255).int().abs()
    x = x.detach().cpu().numpy()
    x = np.array(x, dtype=np.uint8).squeeze(0)
    x = np.transpose(x, [1, 2, 0])
    return x


def train(args, root):
    global log_file
    if not os.path.exists(os.path.join(root, "logs")):
        os.mkdir(os.path.join(root, "logs"))
    if not os.path.exists(os.path.join(root, "logs/result/")):
        os.mkdir(os.path.join(root, "logs/result/"))
    if not os.path.exists(os.path.join(root, "logs/result/event")):
        os.mkdir(os.path.join(root, "logs/result/event"))
    log_file = open(os.path.join(root, "logs/log.txt"), "w")
    to_log(args)
    writer = SummaryWriter(os.path.join(root, "logs/result/event/"))

    args_data = args['data']
    args_train = args['train']
    args_model = args['model']
    loss_lambda = args['loss_lambda']

    dataloader = build_data(args_data['data_tag'], args_data['data_path'], args_train["bs"], True,
                            num_worker=args_train["num_workers"], classes=args_model['num_classes'],
                            image_size=args_train['image_size'])
    E = vae_encoder(3, args_model['output_length'], args_model['depth'], args_train['image_size'],
                    args_model['num_classes']).cuda()
    D = vae_decoder(3, args_model['output_length'], args_model['depth'], args_train['image_size'],
                    args_model['num_classes']).cuda()
    E_opt = torch.optim.Adam(E.parameters(), lr=args_train["lr"], betas=(0.5, 0.9))
    D_opt = torch.optim.Adam(D.parameters(), lr=args_train["lr"], betas=(0.5, 0.9))
    E_sch = torch.optim.lr_scheduler.MultiStepLR(E_opt, args_train["lr_milestone"], gamma=0.5)
    D_sch = torch.optim.lr_scheduler.MultiStepLR(D_opt, args_train["lr_milestone"], gamma=0.5)

    load_epoch = load({"E": E, "D": D, "E_opt": E_opt, "D_opt": D_opt, "E_sch": E_sch, "D_sch": D_sch},
                      args_train["load_epoch"], root)
    tot_iter = (load_epoch + 1) * len(dataloader)

    E_opt.step()
    D_opt.step()
    criterion = vae_loss
    for epoch in range(load_epoch + 1, args_train['epoch']):
        E_sch.step()
        D_sch.step()
        for i, (image, label) in enumerate(dataloader):
            E_opt.zero_grad()
            D_opt.zero_grad()
            tot_iter += 1
            image = image.cuda()
            label = label.cuda()
            mu, log_var = E(image, label)
            y = D(E.reparameterize(mu, log_var), label)
            losses = criterion(image, y, mu, log_var, loss_lambda)
            losses['total'].backward()
            E_opt.step()
            D_opt.step()
            if tot_iter % args_train['show_interval'] == 0:
                to_log(
                    'epoch: {}, batch: {}, lr: {}'.format(
                        epoch, i, E_sch.get_last_lr()[0]), end=' ')
                for key in losses.keys():
                    to_log(', ' + key + ": {}".format(losses[key].item()), end=" ")
                to_log('')
            for key in losses.keys():
                writer.add_scalar('loss/' + key, losses[key], tot_iter)
            writer.add_scalar("lr", E_sch.get_last_lr()[0], tot_iter)

        if epoch % args_train["snapshot_interval"] == 0:
            torch.save(E.state_dict(), os.path.join(root, "logs/E_epoch-{}.pth".format(epoch)))
            torch.save(D.state_dict(), os.path.join(root, "logs/D_epoch-{}.pth".format(epoch)))
            torch.save(E_opt.state_dict(), os.path.join(root, "logs/E_opt_epoch-{}.pth".format(epoch)))
            torch.save(D_opt.state_dict(), os.path.join(root, "logs/D_opt_epoch-{}.pth".format(epoch)))
            torch.save(E_sch.state_dict(), os.path.join(root, "logs/E_sch_epoch-{}.pth".format(epoch)))
            torch.save(D_sch.state_dict(), os.path.join(root, "logs/D_sch_epoch-{}.pth".format(epoch)))
        if epoch % args_train['test_interval'] == 0:
            # label = torch.tensor([5]).expand([64]).cuda()
            # input_test[:, 1, :, :] = label.reshape(64, 1, 1).expand(64, image_size, image_size)
            # G_out = G(input_test)
            image = y.clone().detach()
            image = batch_image_merge(image)
            image = imagetensor2np(image)
            y = y / 2 + 0.5
            y = y.clamp(0, 1)
            save_image(y, os.path.join(root, "logs/reconstruction-{}.png".format(epoch)))
            writer.add_image('image{}/fake'.format(epoch), cv2.cvtColor(image, cv2.COLOR_BGR2RGB), tot_iter,
                             dataformats='HWC')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--test", default=False, action='store_true')
    args = parser.parse_args()
    train(open_config(args.root), args.root)
