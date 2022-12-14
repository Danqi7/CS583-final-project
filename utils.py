    # Visualize tanh matrix/edge weights
    vid = -1 #last
    edge_inds, edge_ws = relation_s[-1]
    edge_inds = edge_inds.cpu().detach().numpy()
    edge_ws = edge_ws.cpu().detach().numpy()
    v_adj = np.zeros((node_degree, node_degree)) # [N, N]
    v_adj[edge_inds[0,:], edge_inds[1, :]] = np.squeeze(edge_ws)
    plt.matshow(v_adj)
    plt.savefig(dir_name + '/relation_viz.png')
