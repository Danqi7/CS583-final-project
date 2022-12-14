import numpy as np

from torch.utils.data import Dataset

# node_degree=600

class CellClusterData(Dataset):
    def __init__(self, root_dir, feat_dim, chunk_len, node_degree, type, adj_mat=None, edge_index=None):
        self.root_dir = root_dir
        self.type = type
        self.feat_dim = feat_dim
        self.chunk_len = chunk_len
        self.adj_mat = adj_mat
        self.edge_index = edge_index
        
        self.data = np.load(root_dir) #[D, T, N]

        self.t_bucket = self.data.shape[1] // chunk_len
        self.cell_groups = int(np.floor(self.data.shape[0] / feat_dim))
        self.train_cells = int(self.cell_groups * 0.7)
        self.test_cells = self.cell_groups - self.train_cells

        self.train_nums = self.train_cells * self.t_bucket
        self.test_nums = self.test_cells * self.t_bucket

        self.adj_method = 'random'
        if adj_mat is None:
          self.adj_mat, self.edge_index = self.adjacency_matrix(method='random', nums=node_degree)
    
    def adjacency_matrix(self, method='random', nums=100):
      nodes_n = self.data[0].shape[-1]
      adj = np.zeros((nodes_n, nodes_n)) # [N, N]
      edge_index = np.empty((2,0)) # [2, |E|]

      if method == 'random':
        for i in range(nodes_n):
          adj_inds = np.random.choice(nodes_n, nums, replace=False)
          adj[i, adj_inds] = 1
          
          node_edge_index = np.vstack((np.array([i]*nums), adj_inds))
          edge_index = np.hstack((edge_index, node_edge_index))
      
      return adj, edge_index

    def __len__(self):
        if self.type == 'train':
          return self.train_nums
        else:
          return self.test_nums

    def __getitem__(self, index):
        #print('index: ... ', index)
        # Get specific group from [D, T, N]
        #print(self.data.shape)
        cell_group_index = int(np.floor(index / self.t_bucket))
        time_group_index = index % self.t_bucket

        if self.type == 'train':
          csi = int(cell_group_index)
          cei = int(cell_group_index + self.feat_dim)
          tsi = int(time_group_index)
          tei = int(time_group_index + self.chunk_len)
          #print('si: ', si, 'ei: ', ei, self.data[si:ei, :, :].shape)
          return self.data[csi:cei, tsi:tei :]
        else:
          csi = cell_group_index + self.train_cells
          cei = csi + self.feat_dim
          tsi = time_group_index
          tei = time_group_index + self.chunk_len
          return self.data[csi:cei, tsi:tei :]

    def __str__(self):
        return 'data type: %s\n  of data (D, T, N): %s, edge_index: %s, train_nums: %d' % ('cell cluster '+self.type, 
                                                                         str(self.data.shape), 
                                                                         str(self.edge_index.shape),
                                                                         self.train_nums)
