from __future__ import print_function
import argparse
from utils.utils import *
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from data_loader.get_loader import get_loader
import numpy as np
import os
import matplotlib.pyplot as plt
from operator import xor
import time
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Openset DA')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--net', type=str, default='resnet152', metavar='B',
                    help='which network alex,vgg,res?')
parser.add_argument('--save', action='store_true', default=False,
                    help='save model or not')
parser.add_argument('--save_path', type=str, default='checkpoint/', metavar='B',
                    help='checkpoint path')
parser.add_argument('--source_path', type=str, default='./utils/source_list.txt', metavar='B',
                    help='checkpoint path')
parser.add_argument('--target_path', type=str, default='./utils/target_list.txt', metavar='B',
                    help='checkpoint path')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--unit_size', type=int, default=1000, metavar='N',
                    help='unit size of fully connected layer')
parser.add_argument('--update_lower', action='store_true', default=False,
                    help='update lower layer or not')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable cuda')
parser.add_argument('--model_path', type=str,metavar='B',
                    help='Model path')
parser.add_argument('--train', action='store_true', default=False,
                    help='Train the model')
parser.add_argument('--test', action='store_true', default=False,
                    help='Test the model')
parser.add_argument('--dataset', type=str,metavar='B',
                    help='dataset name')
args = parser.parse_args()

plt.rcParams.update({'font.size':20})
def Plot_confusionMatrix(y_true, y_pred, figname):
    classes = ['Baseball Field', 'Beach', 'Medium Residential', 'Parking Lot', 'Sparse Residential', 'Unknown']
    m = confusion_matrix(y_true, y_pred)
    m = (m.T/np.sum((m.T), axis=0)).T
    m = np.around(m, 2)
    df_cm = pd.DataFrame(m, classes, classes)
    fig, ax = plt.subplots(figsize=(9,6))
    sns.heatmap(df_cm, annot=True, fmt=".2g", cmap='Blues')
    for label in ax.get_xticklabels():
        #label.set_ha("right")
        label.set_rotation(0)
    for label in ax.get_yticklabels():
        label.set_rotation(45)
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    ax.set_xticklabels(xticklabels, fontsize=7)
    ax.set_yticklabels(yticklabels, fontsize=7)
    plt.tight_layout()
    plt.savefig(figname, dpi=100)

class FeatAccumulator:
    def __init__(self):
        self.features = []
    def collect(self, feat_in):
        self.features.append(feat_in.detach().cpu())
        return self
    def ToArray(self):
        return torch.cat(self.features).numpy()

def ApplyTSNE(data):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    return tsne_results

def TSNE_plot(results, y_true, figname):
    #plt.figure(figsize=(16,10))
    sns.scatterplot(
            x=results[:,0], y=results[:,1],
        hue=y_true,
        palette=sns.color_palette("hls", 6),
        legend="full",
        )
    plt.savefig(figname, dpi=100)

def Failure_images(files, true, pred):
    if not isinstance(files, np.ndarray):
        files = np.array(files)
    knn_labels = np.logical_and(true<5, true!=pred)
    unkn_labels = np.logical_and(true >4, pred!=5)
    labels = np.logical_or(knn_labels, unkn_labels)
    return [[item[0],item[1],item[2]] for item in zip(files[labels], true[labels], pred[labels])]

def Dump_text(files, filename):
    with open(filename,'w') as fw:
        for item in files:
            fw.write(item[0] + ' ' + str(item[1]) + ' ' + str(item[2]) + '\n')

def Load_txt(img_paths):
    with open(img_paths,'rb') as fr:
        image_paths = [img_path.split()[0].decode("utf-8") for img_path in fr.readlines()]
    return image_paths

args.cuda = not args.no_cuda and torch.cuda.is_available()
assert xor(args.train, args.test)
source_data = args.source_path
target_data = args.target_path
evaluation_data = args.target_path
batch_size = args.batch_size
data_transforms = {
    source_data: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    target_data: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    evaluation_data: transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
train_loader, test_loader = get_loader(source_data, target_data, evaluation_data,
                                       data_transforms, batch_size=args.batch_size)
dataset_train = train_loader.load_data()
dataset_test = test_loader

if args.dataset == 'VISDA':
    num_class = 7
    class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]
elif args.dataset in ['UCM', 'AID']:
    num_class = 6
    class_list = ["baseballdiamond", "beach", "mediumresidential", "parkinglot", \
           "sparseresidential", "unkn"]
else:
    raise NotImplementedError('Dataset not implemented. Use VISDA or UCM or AID')
# ['airplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle',
# 'person', 'plant', 'skateboard', 'train', 'truck', 'unk']
G, C = get_model(args.net, num_class=num_class, unit_size=args.unit_size)
if args.cuda:
    G.cuda()
    C.cuda()
opt_c, opt_g = get_optimizer_visda(args.lr, G, C,
                                   update_lower=args.update_lower)

print(args.save_path)


def train(num_epoch):
    criterion = nn.CrossEntropyLoss().cuda()
    i = 0
    print('train start!')
    for ep in range(num_epoch):
        G.train()
        C.train()
        for batch_idx, data in enumerate(dataset_train):
            i += 1
            if i % 1000 == 0:
                print('iteration %d', i)
            if args.cuda:
                img_s = data['S']
                label_s = data['S_label']
                img_t = data['T']
                img_s, label_s = Variable(img_s.cuda()), \
                                 Variable(label_s.cuda())
                img_t = Variable(img_t.cuda())
            if len(img_t) < batch_size:
                break
            if len(img_s) < batch_size:
                break
            opt_g.zero_grad()
            opt_c.zero_grad()
            feat = G(img_s)
            out_s = C(feat)
            loss_s = criterion(out_s, label_s)
            loss_s.backward()
            target_funk = Variable(torch.FloatTensor(img_t.size()[0], 2).fill_(0.5).cuda())
            p = 1.0
            C.set_lambda(p)
            feat_t = G(img_t)
            out_t = C(feat_t, reverse=True)
            out_t = F.softmax(out_t)
            prob1 = torch.sum(out_t[:, :num_class - 1], 1).view(-1, 1)
            prob2 = out_t[:, num_class - 1].contiguous().view(-1, 1)

            prob = torch.cat((prob1, prob2), 1)
            loss_t = bce_loss(prob, target_funk)
            loss_t.backward()
            opt_g.step()
            opt_c.step()
            opt_g.zero_grad()
            opt_c.zero_grad()

            if batch_idx % args.log_interval == 0:
                print('Train Ep: {} [{}/{} ({:.0f}%)]\tLoss Source: {:.6f}\t Loss Target: {:.6f}'.format(
                    ep, batch_idx * len(data), 70000,
                        100. * batch_idx / 70000, loss_s.item(), loss_t.item()))
            if ep > 0 and batch_idx % 1000 == 0:
                test()
                G.train()
                C.train()

        if args.save:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            save_model(G, C, args.save_path + 'checkpoint_' + str(ep))


def test(load=False):
    if load:
        load_model(G, C, args.model_path)
    G.eval()
    C.eval()
    correct = 0
    size = 0
    per_class_num = np.zeros((num_class))
    per_class_correct = np.zeros((num_class)).astype(np.float32)
    all_pred = []
    all_true = []
    acc_tsne_fs = FeatAccumulator()
    for batch_idx, data in enumerate(dataset_test):
        if args.cuda:
            img_t, label_t, path_t = data[0], data[1], data[2]
            img_t, label_t = Variable(img_t.cuda(), volatile=True), \
                             Variable(label_t.cuda(), volatile=True)
        feat = G(img_t)
        out_t = C(feat)
        pred = out_t.data.max(1)[1]
        k = label_t.data.size()[0]
        correct += pred.eq(label_t.data).cpu().sum()
        pred = pred.cpu().numpy()
        all_pred.append(pred)
        all_true.append(label_t.cpu().numpy())
        acc_tsne_fs.collect(feat)
        for t in range(num_class):
            t_ind = np.where(label_t.data.cpu().numpy() == t)
            correct_ind = np.where(pred[t_ind[0]] == t)
            per_class_correct[t] += float(len(correct_ind[0]))
            per_class_num[t] += float(len(t_ind[0]))
        size += k
    per_class_acc = per_class_correct / per_class_num
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    target_list = Load_txt(target_data)
    #Dump_text(Failure_images(target_list, y_true, y_pred), 'opda_failure.txt')
    print(
        '\nTest set including unknown classes:  Accuracy: {}/{} ({:.0f}%)  ({:.4f}%)\n'.format(
            correct, size,
            100. * correct / size, float(per_class_acc.mean())))
    for ind, category in enumerate(class_list):
        print('%s:%s' % (category, per_class_acc[ind]))
    #df = ApplyTSNE(acc_tsne_fs.ToArray())
    #TSNE_plot(df, y_true, "opda_wda_tsne_df.png")
    Plot_confusionMatrix(y_true, y_pred, 'opda_wda_confmatrix_blues_lf.png')

if args.train:
    train(args.epochs + 1)
if args.test:
    test(True)
