from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from visdom import Visdom
import numpy as np
from PIL import Image
import gc

# Training settings
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--imageSize', type=tuple, default=(64,64),
                     help='the height / width of the input image to network')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='M',
                    help='SGD weight_decay (default: 0.0005)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log_interval_epoch', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log_interval_epoch_test', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.5, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')
parser.add_argument('--con_embed', default=0.2, metavar='M',
                    help='constant embedded (default:0.2)')

best_acc = 0



def default_image_loader(path):
    return Image.open(path)

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, filenames_filename, triplets_file_name, transform=None,
                 loader=default_image_loader):
        """ image.txt: A text file with each line containing the path to an image e.g.,
                .../train/timage/sample.jpg (image_size: 64 * 64)
            triplet.txt: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image b than it is to image c, e.g., 
                0 1 42 """
        self.base_path = base_path  
        self.filenamelist = []
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        triplets = []
        for line in open(triplets_file_name):
            triplets.append((line.split()[0], line.split()[1], line.split()[2])) # anchor, far, close
        self.triplets = triplets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets[index]
        img1 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path1)]))
        img2 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path2)]))
        img3 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path3)]))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)

class Net(nn.Module):
    def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
          self.conv2 = nn.Conv2d(20, 20, kernel_size=3)
          self.conv2_drop = nn.Dropout2d()
          self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
          self.conv4 = nn.Conv2d(40, 40, kernel_size=3)
          self.conv4_drop = nn.Dropout2d() 

    def forward(self, x):
          x = F.max_pool2d(F.relu(self.conv1(x)), 2)
          x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))),3, 2)
          x = F.max_pool2d(F.relu(self.conv3(x)), 2)
          x = F.max_pool2d(F.relu(self.conv4_drop(self.conv4(x))), 2)

          x = x.view(-1, 160)
          return x
    
class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
          super(Tripletnet, self).__init__()
          self.embeddingnet = embeddingnet
 
    def forward(self, x, y, z):
          embedded_x = self.embeddingnet(x)
          embedded_y = self.embeddingnet(y)
          embedded_z = self.embeddingnet(z)
          dist_a = F.pairwise_distance(embedded_x, embedded_z, 2)
          dist_b = F.pairwise_distance(embedded_x, embedded_y, 2)
          return dist_a, dist_b, embedded_x, embedded_y, embedded_z


def main(datastr,scale,step):
    accR = []
    lossR = []
    lossRT = []
    netRecord = []
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


    trainImage = datastr + '/train/tripletimage/image.txt'
    trainTriplet = datastr + '/train/tripletimage/triplet.txt'
        
    testImage5 = datastr + '/test/tripletimage/image.txt'
    testTriplet5 = datastr + '/test/tripletimage/triplet.txt'


    

    
    train_loader = torch.utils.data.DataLoader(
            TripletImageLoader('../data',trainImage,
                               trainTriplet,
                               transform=transforms.Compose([
                               transforms.ToTensor()
                               ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
       
    test_loader5 = torch.utils.data.DataLoader(
            TripletImageLoader('../data', testImage5,
                               testTriplet5,
                               transform=transforms.Compose([
                               transforms.ToTensor()
                               ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    

   


    model = Net()
    ###
    if step > 0:        
        modelstr = datastr + '/model/model_' + str((scale-1)*50 + step)
        modelstr = modelstr + '.pkl'
        model.load_state_dict(torch.load(modelstr))

    ###
    tnet = Tripletnet(model)
    if args.cuda:
        tnet.cuda()


    cudnn.benchmark = True

    criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    optimizer = optim.SGD(tnet.parameters(), lr=args.lr,weight_decay = args.weight_decay, momentum=args.momentum)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))


    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        losst = train(train_loader, tnet, criterion, optimizer, epoch,netRecord)
        lossRT.append(losst)
        # evaluate on validation set
        if epoch % args.log_interval_epoch_test == 0:

            tempAcc = []
            tempLoss = []
            acc,loss = test(test_loader5, tnet, criterion, epoch,netRecord)
            tempAcc.append(acc*100)
            tempLoss.append(loss)

            
            accR.append(tempAcc)
            lossR.append(tempLoss)



        if epoch % args.log_interval_epoch == 0:
            modelstr = datastr + '/model/model_'+ str(epoch*scale + step)
            modelstr = modelstr + '.pkl'
            torch.save(model.state_dict(), modelstr)
            
            recordstr = datastr + '/rate/model_'+ str(epoch*scale + step)+'_' + 'modelRecord'
            recordstr = recordstr + '.txt'
            frecordstr = open(recordstr, 'w')
            
            lossstr = datastr + '/rate/model_'+ str(epoch*scale + step) + '_' + 'lossTrain'
            lossstr = lossstr + '.txt'
            flossTrain = open(lossstr, 'w')
            accstr = datastr + '/rate/model_'+ str(epoch*scale + step) + '_' + 'accTest'
            accstr = accstr + '.txt'
            faccTest = open(accstr, 'w')
            lossstr = datastr + '/rate/model_'+ str(epoch*scale + step) + '_' + 'lossTest'
            lossstr = lossstr + '.txt'
            flossTest = open(lossstr, 'w')
            
            
                
            for i in range(len(lossRT)):
                flossTrain.write(str(lossRT[i]))
                flossTrain.write('\n')
            flossTrain.close()
            
            for i in range(len(netRecord)):
                frecordstr.write(netRecord[i])
                frecordstr.write('\n')
            frecordstr.close()
            
            for i in range(len(accR)):
                acc_list = accR[i]
                loss_list = lossR[i]
                for j in range(len(acc_list)):
                    faccTest.write(str(acc_list[j]))
                    faccTest.write(' ')
                    flossTest.write(str(loss_list[j]))
                    flossTest.write(' ')
                faccTest.write('\n')
                flossTest.write('\n')
            faccTest.close()
            flossTest.close()
        gc.collect()

def train(train_loader, tnet, criterion, optimizer, epoch,netRecord):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda() 
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet 

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet.data[0], data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
            
        netRecord.append('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
    return losses.avg


def test(test_loader, tnet, criterion, epoch,netRecord):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, _, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        test_loss =  criterion(dista, distb, target).data[0]

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    netRecord.append('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    return accs.avg,losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

if __name__ == '__main__':
#    step = 0
    datastr = 'G:/github/data/'

    step = 0
    i = 1
    main(datastr,i,step)

