import torch

def eval_test(model, loss_fn, edge_index, test_loader, test_data, device):
    loss_total = 0
    pred_traj = []
    target_traj = []
    with torch.no_grad():
      edge_index = torch.Tensor(edge_index).type(torch.LongTensor).to(device)      
      for _, x in enumerate(test_loader, 0):
        x = torch.squeeze(x, 0) # [B, D, T, N] -> [D, T, N]
        print('x.shape: ', x.shape)
        x = torch.permute(x, (1, 2, 0)) # [T, N, D]
        x = x.type(torch.FloatTensor).to(device)
        
        #print(x.type())

        x_t0 = x[0, :, :]
        target = x[1:, :, :] # autoregress T-1 seq
        #print('steps: target.shape[0]', target.shape[0])
        x_s, attn_s, relation_s = model(x_t0, edge_index, steps=target.shape[0])
        pred_traj.append(torch.cat((torch.unsqueeze(x_t0, 0), x_s), dim=0))
        target_traj.append(torch.cat((torch.unsqueeze(x_t0, 0), target), dim=0))


        #print('x_s, target: ', x_s.shape, target.shape)
        
        loss = loss_fn(x_s, target) # [T-1,]
        loss = torch.mean(loss, 1) #[T-1, feat_dim] avg over all genes
        loss = torch.mean(loss, 0) # [feat_dim] avg over all timesteps
        loss = torch.mean(loss) # avg over dims
        loss_total += loss.item()

        # target = torch.permute(target, (0, 2, 1)) # [T, N, D]
        # x_t0 = torch.unsqueeze(torch.Tensor(x_t0), 0)
        # x_t0 = torch.squeeze(torch.permute(x_t0, (0, 2, 1)), 0) # [T, N, D]
        # print(target.shape, x_t0.shape)
        # target = target.type(torch.FloatTensor).to(device)
        # x_t0 = x_t0.type(torch.FloatTensor).to(device)
        # #print(x.type())

        # # x_t0 = x[0, :, :]
        # #target = x[1:, :, :] # autoregress T-1 seq
        
        # edge_index = torch.Tensor(edge_index).type(torch.LongTensor).to(device)
        # print(edge_index.shape)
        # x_s, attn_s = model(x_t0, edge_index, steps=target.shape[0])
        # print('x_s, target: ', x_s.shape, target.shape)
        # loss = loss_fn(x_s, target) # [T-1,]


    return loss_total / len(test_data), pred_traj, target_traj