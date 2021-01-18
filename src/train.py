import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model import ProtoNet
from parser import get_parser
from torchvision import transforms
from data_utils import BatchSampler
from torch.utils.data import DataLoader
from omiglot_dataset import OmniglotDataset
from proto_loss import ProtoLoss as criterion
from miniImagenet_dataset import MiniImagenetDataset


def init_seed(opt):
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_dataset(opt, mode):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])
    if opt.dataset_type == "miniImagenet":
        dataset = MiniImagenetDataset(opt.dataset_root, mode, transform)
    else:
        dataset = OmniglotDataset(opt.dataset_type, mode)
    n_classes = len(np.unique(dataset.y))
    if (mode == "train" and n_classes < opt.class_per_epi_tr) or (mode == "val" and n_classes < opt.class_per_epi_val):
        raise (Exception('There are not enough classes in the dataset in order to satisfy the chosen classes_per_it. ' +
                         'Decrease the classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if mode == "train":
        classes_per_eps = opt.class_per_epi_tr
        num_samples = opt.num_spt_tr + opt.num_qry_tr
    else:
        classes_per_eps = opt.class_per_epi_val
        num_samples = opt.num_spt_val + opt.num_qry_val

    return BatchSampler(labels=labels,
                        classes_per_eps=classes_per_eps,
                        num_samples=num_samples,
                        episodes=opt.episodes)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_model():
    model = nn.DataParallel(ProtoNet())
    return model.cuda()


def init_optim_scheduler(opt, model):
    optim = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                gamma=opt.lr_scheduler_gamma,
                                                step_size=opt.lr_scheduler_step)
    return optim, scheduler


def train(opt, tr_dataloader, model, optim, scheduler, val_dataloader):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    best_state = None

    best_model_path = os.path.join(opt.model_out, opt.dataset_type, "best_model.pth")
    last_model_path = os.path.join(opt.model_out, opt.dataset_type, "last_model.pth")

    for epoch in range(opt.epochs):
        print("== Epoch {} is Training ==".format(epoch))
        model.train()
        optim.zero_grad()

        for batch in tqdm(tr_dataloader):
            x, y = batch
            x, y = x.cuda(), y.cuda()
            output = model(x)
            loss, acc = criterion(output, y, opt.num_spt_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.episodes])
        avg_acc = np.mean(train_acc[-opt.episodes])
        print("Avg Train Loss: {}, Avg Train Acc: {}.".format(avg_loss, avg_acc))
        scheduler.step()
        if val_dataloader is None:
            continue

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                x, y = batch
                x, y = x.cuda(), y.cuda()
                output = model(x)
                loss, acc = criterion(output, y, opt.num_spt_val)
                val_loss.append(loss.item())
                val_acc.append(acc.item())
            avg_loss = np.mean(val_loss[-opt.episodes])
            avg_acc = np.mean(val_acc[-opt.episodes])
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
            print("Avg Val loss: {}, Avg Val acc: {} {}".format(avg_loss, avg_acc, postfix))

            if avg_acc >= best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)
    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        with open(name+".txt", "w", encoding="utf-8") as f:
            for item in locals()[name]:
                f.write("%s\n" % item)

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    model.eval()
    avg_acc = []
    for epoch in range(10):
        for batch in tqdm(test_dataloader):
            x, y = batch
            x, y = x.cuda(), y.cuda()
            output = model(x)
            _, acc = criterion(output, y, opt.num_spt_tr)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print("Test Acc: {}".format(avg_acc))
    return avg_acc


def main():
    options = get_parser().parse_args()

    if not os.path.exists(os.path.join(options.model_out, options.dataset_type)):
        os.makedirs(os.path.join(options.model_out, options.dataset_type))

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader = init_dataloader(options, "train")
    val_dataloader = init_dataloader(options, "val")
    test_dataloader = init_dataloader(options, "test")
    model = init_model()
    optim, scheduler = init_optim_scheduler(options, model)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = train(options, tr_dataloader, model, optim, scheduler, val_dataloader)

    print('Testing with last model..')
    test(options, test_dataloader, model)

    print('Testing with best model..')
    model.load_state_dict(best_state)
    test(options, test_dataloader, model)


if __name__ == '__main__':
    main()
