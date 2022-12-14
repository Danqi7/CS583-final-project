import os
import time
import argparse
import matplotlib.pyplot as plt

import numpy as np
from tqdm import trange, tqdm # used for fancy status bars

import torch
import torch.nn.functional as F

from datasets import CellClusterData
from models import GResidule
from eval import eval_test


# Set the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train_loader, edge_index, model, loss_fn, optimizer, epoch, device, test_loader, test_data):
    edge_index = torch.Tensor(edge_index).type(torch.LongTensor).to(device)
    print(device, edge_index.shape, edge_index.type())
    # data: [T, D, N], edge_index is fixed [2, |E|]

    model.train()
    losses = []
    test_losses = []

    for e in trange(epoch):
      for _, x in enumerate(train_loader, 0):
        optimizer.zero_grad()

        x = torch.squeeze(x, 0) # [B, D, T, N] -> [D, T, N]
        #print('x.shape: ', x.shape)
        x = torch.permute(x, (1, 2, 0)) # [T, N, D]
        x = x.type(torch.FloatTensor).to(device)
        #print(x.type())

        x_t0 = x[0, :, :]
        target = x[1:, :, :] # autoregress T-1 seq
        #print('steps: target.shape[0]', target.shape[0])
        x_s, attn_s, relation_s = model(x_t0, edge_index, steps=target.shape[0])
        #print('x_s, target: ', x_s.shape, target.shape)
        loss = loss_fn(x_s, target) # [T-1,]
        loss = torch.mean(loss, 1) #[T-1, feat_dim] avg over all genes
        loss = torch.mean(loss, 0) # [feat_dim] avg over all timesteps
        loss = torch.mean(loss) # avg over dims
        #print(loss.shape)

        loss.backward()
        optimizer.step()

        #losses.append(loss.item())
      if e % 10 == 0:
        print('[Epoch %d]Loss: %.4f'%(e, loss.item()))
        losses.append(loss.item())
      if e % 100 == 0:
        test_loss, _, _ = eval_test(model, loss_fn, edge_index, test_loader, test_data, device)
        test_losses.append(test_loss)

    
    return losses, test_losses, x_s, attn_s, relation_s




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--store_files", type=str, default='./models/'+ model_name + '/',
    #                     help="Where to store the trained model")
    # parser.add_argument("--batch_size", default=64,
    #                     type=int, help="batch size")
    parser.add_argument("--data", type=str, default='worm',
                        help="worm|eb")
    parser.add_argument("--epochs", default=1600,
                        type=int, help="epochs to run")
    parser.add_argument("--chunk_len", default=10,
                        type=int, help="length of chunck, since too large would cause memory overload")
    parser.add_argument("--lr", default=0.0001,
                        type=float, help="learning rate")
    #parser.add_argument("--save_model", default=False, action="store_true", help="Whether to save model checkpoint")
    args = parser.parse_args()

    # Data
    if args.data == 'worm':
        dpath = './worm_traj_gene_space_50t_60n.npz'
        cells_per_group = 6
        node_degree = 700
    elif args.data == 'eb':
        dpath = './traj_gene_space_50t_500n.npz'
        cells_per_group = 10
        node_degree = 600
    else:
        print('Data Not Supported. ')

    data = CellClusterData(dpath, type='train', feat_dim=cells_per_group,
                           chunk_len=args.chunk_len, node_degree=node_degree)
    test_data = CellClusterData(dpath, type='test', feat_dim=cells_per_group,
                                chunk_len=args.chunk_len, node_degree=node_degree, adj_mat=data.adj_mat, edge_index=data.edge_index)
    
    print(data) # [D, T, N]
    print(test_data)
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False)

    # Hyperparams
    torch.manual_seed(0)
    in_dim = data.feat_dim
    model = GResidule(in_channels=in_dim, out_channels=in_dim).to(device)
    learning_rate = args.lr
    epoch = args.epochs

    loss_fn = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train
    start_time = time.time()
    losses, test_losses, x_s, attn_s, relation_s = train(train_loader, data.edge_index, model, loss_fn, optimizer, epoch, device, test_loader, test_data)
    elapsed_time = time.time() - start_time
    print('Training Finished in : ', time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_time)))
    
    # Clear Cache
    torch.cuda.empty_cache()

    # Create model directory
    gdrive_path = './log/'
    start_time = time.time()
    dir_name = gdrive_path + time.strftime("%m_%d_%H_%M")

    os.mkdir(dir_name)
    print('Saving model to dir: ', dir_name)

    # Save tanh matrix/edge weights
    vid = -1 #last
    edge_inds, edge_ws = relation_s[-1]
    edge_inds = edge_inds.cpu().detach().numpy()
    edge_ws = edge_ws.cpu().detach().numpy()
    
    np.save(dir_name + '/edge_inds.npy', edge_inds)
    np.save(dir_name + '/edge_ws.npy', edge_ws)

    # Plot
    plt.figure()
    plt.title('Train Loss per 10th epoch')
    plt.plot(losses)
    plt.xlabel('ith 10th epoch')
    plt.ylabel('Train Loss')
    plt.savefig(dir_name + '/train_loss.png')

    plt.figure()
    plt.title('Test Loss per 100th epoch')
    plt.plot(test_losses)
    plt.xlabel('ith 100th epoch')
    plt.ylabel('Test Loss')
    plt.savefig(dir_name + '/test_loss.png')

    # Test Loss
    test_loss, pred_traj, target_traj = eval_test(model, loss_fn, data.edge_index, test_loader, test_data, device)
    print('Final test loss: ', test_loss)

    # save some model info
    f = open(dir_name + "/model_info.txt", "a")
    content = "model: " + dir_name + "\n"
    content += "\nData: " + str(args.data)
    content += "\nNode degree: " + str(node_degree)
    content += "\nTrain Loss: " + str(losses[-1])
    content += "\nTest Loss: " + str(test_loss)
    content += "\nlr: " + str(learning_rate) + "\nbatch size: " + str(batch_size) + "\nnum_epochs: " + str(epoch)
    content += "\nArchitecture: " + model.__str__()
    f.write(content)
    f.close()
