import torch

from RegulateConv import RegulateConv

class GResidule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.conv1 = RegulateConv(in_channels, out_channels, heads, return_attention_weights=True)

        # self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
        #                      concat=False, dropout=0.6)

    def forward(self, x, edge_index, steps):
        # x: [N x D]

        x_s = [] # [T, N, D]
        attn_s = [] # [T, |E|x|E|]
        relation_s = [] 


        for t in range(steps):
            #x_prime, attn_t = self.conv1(x, edge_index)
            x_prime, attn_t, relation_t = self.conv1(x, edge_index, return_attention_weights=True)
            #print('x_prime: , x: ', x_prime.shape, x.shape)
            #print(attn_t[0].shape, attn_t[1].shape)

            x = x + x_prime
            #print('before append: ', x.shape)
            x_s.append(torch.unsqueeze(x, 0))
            attn_s.append(attn_t)
            relation_s.append(relation_t)

        x_s = torch.cat(x_s, 0)

        return x_s, attn_s, relation_s