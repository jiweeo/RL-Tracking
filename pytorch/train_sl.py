import argparse
from torchvision import transforms
import Dataset
from utils import AverageMeter
from torch.utils.data import DataLoader
import Network
import torch
import time
from torch.nn import functional as F
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--train_data', type=str, default='../data/train/')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_data', type=str, default='../data/test/')
    parser.add_argument('--test_freq', type=int, default=5)
    parser.add_argument('--resume', type=str, default='',  help='path to checkpoint (default: none)')
    parser.add_argument('--load', type=str, default='', help='path to checkpoint(default: none)')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--name', type=str, default='run')
    return parser.parse_args()


def get_transform():
    transform = transforms.Compose([
        transforms.Resize((114, 114)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def train(epoch, net, optimzer, train_loader):
    # setting metrics
    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cls_losses = AverageMeter()
    action_losses = AverageMeter()
    action_accs = AverageMeter()
    cls_accs = AverageMeter()

    net.train()

    for i, (input, target_action, target_cls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input, target_cls, target_action = input.cuda(), target_cls.cuda(), target_action.cuda()

        # run model
        action, cls = net(input)

        # cal loss
        action_loss = F.cross_entropy(action, target_action)
        cls_loss = F.cross_entropy(cls, target_cls)
        action_losses.update(action_loss)
        cls_losses.update(cls_loss)
        loss = action_loss + cls_loss

        # update net
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        # cal acc
        action = torch.argmax(action, 1)
        cls = torch.argmax(cls, 1)
        correct_action = (action == target_action).cpu().sum()
        correct_cls = (cls == target_cls).cpu().sum()
        action_acc = float(correct_action) / input.shape[0]
        cls_acc = float(correct_cls) / input.shape[0]
        action_accs.update(action_acc)
        cls_accs.update(cls_acc)

        batch_time.update(time.time()-end)
        end = time.time()

        if i % args.log_freq == 0:
            if i % args.log_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Action_Loss {action_loss.val:.4f} ({action_loss.avg:.4f})\t'
                      'Class_Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'action_acc {action_acc.val:.4f} ({action_acc.avg:.4f})\t'
                      'cls_acc {cls_acc.val:.4f} ({cls_acc.avg:.4f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    action_loss=action_losses, cls_loss=cls_losses,
                    action_acc=action_accs, cls_acc=cls_accs))


def test(epoch, net, test_loader):
    end = time.time()
    batch_time = AverageMeter()
    action_accs = AverageMeter()
    cls_accs = AverageMeter()

    net.eval()
    for i, (input, target_action, target_cls) in enumerate(test_loader):
        input, target_cls, target_action = input.cuda(), target_cls.cuda(), target_action.cuda()

        # run model
        action, cls = net(input)

        # cal acc
        action = torch.argmax(action, 1)
        cls = torch.argmax(cls, 1)
        correct_action = (action == target_action).cpu().sum()
        correct_cls = (cls == target_cls).cpu().sum()
        action_acc = float(correct_action) / input.shape[0]
        cls_acc = float(correct_cls) / input.shape[0]
        action_accs.update(action_acc)
        cls_accs.update(cls_acc)

        batch_time.update(time.time()-end)
        end = time.time()

        if i % args.log_freq == 0:
            if i % args.log_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'action_acc {action_acc.val:.4f} ({action_acc.avg:.4f})\t'
                      'cls_acc {cls_acc.val:.4f} ({cls_acc.avg:.4f})'.format(
                    epoch, i, len(test_loader), batch_time=batch_time,
                    action_acc=action_accs, cls_acc=cls_accs))
        if i>500:
            break

    return action_accs.avg


def save_checkpoint(state, is_best):
    directory = "cv/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, 'epoch-{}.pth'.format(state['epoch']))
    torch.save(state, directory + 'latest.pth')
    if state['epoch'] % 10 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, directory + 'model_best.pth')


def main():
    global args
    global best_acc
    best_acc = 0
    args = parse_arguments()
    transfrom = get_transform()
    train_dataset = Dataset.TrackData(args.train_data, transfrom)
    test_dataset = Dataset.TrackData(args.test_data, transfrom)
    train_loader = DataLoader(train_dataset,  num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size)

    net = Network.Tracknet()
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), args.lr)

    for e in range(1, args.max_epochs):
        train(e, net, optimizer, train_loader)
        if e % args.test_freq == 0:
            acc = test(e, net, test_loader)
            is_best = acc > best_acc
            best_acc = max(best_acc, acc)
            # save model
            save_checkpoint({
                'epoch': e + 1,
                'state_dict': net.state_dict(),
                'best_accuracy': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best)


if __name__ == '__main__':
    main()
