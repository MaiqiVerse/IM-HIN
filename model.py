import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json


class GATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, dropout=0.0, attn_dropout=0.0,
                 negative_slope=0.2, residual=False, activation=None,
                 allow_zero_in_degree=False, bias=True, *args, **kwargs
                 ):
        r"""
        Graph attention layer from `Graph Attention Network
        <https://arxiv.org/pdf/1710.10903.pdf>`__

        .. math::
            h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{ij} W^{(l)} h_j^{(l)}

        where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
        node :math:`j`:

        .. math::
            \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

            e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

        Parameters
        ----------
        in_feats : int, or pair of ints
            Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
            GATConv can be applied on homogeneous graph and unidirectional
            `bipartite graph.
            If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
            specifies the input feature size on both the source and destination nodes.  If
            a scalar is given, the source and destination node feature size would take the
            same value.
        out_feats : int
            Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
        num_heads : int
            Number of heads in Multi-Head Attention.
        feat_drop : float, optional
            Dropout rate on feature. Defaults: ``0``.
        attn_drop : float, optional
            Dropout rate on attention weight. Defaults: ``0``.
        negative_slope : float, optional
            LeakyReLU angle of negative slope. Defaults: ``0.2``.
        residual : bool, optional
            If True, use residual connection. Defaults: ``False``.
        activation : callable activation function/layer or None, optional.
            If not None, applies an activation function to the updated node features.
            Default: ``None``.
        allow_zero_in_degree : bool, optional
            If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
            since no message will be passed to those nodes. This is harmful for some applications
            causing silent performance regression. This module will raise a DGLError if it detects
            0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
            and let the users handle it by themselves. Defaults: ``False``.
        bias : bool, optional
            If True, learns a bias term. Defaults: ``True``.

        Note
        ----
        Zero in-degree nodes will lead to invalid output value. This is because no message
        will be passed to those nodes, the aggregation function will be appied on empty input.
        A common practice to avoid this is to add a self-loop for each node in the graph if
        it is homogeneous, which can be achieved by:

        >>> g = ... # a scipy sparse matrix
        >>> g = g + sp.eye(g.shape[0])
        >>> g = g.eliminate_zeros()
        >>> indices = torch.from_numpy(np.asarray([g.row, g.col]))
        >>> values = torch.from_numpy(g.data)
        >>> g = torch.sparse_coo_tensor(indices, values, g.shape)

        Examples
        --------
        >>> import numpy as np
        >>> import torch
        >>> import scipy.sparse as sp
        >>> row_indices = np.array([0,1,2,3,2,5])
        >>> col_indices = np.array([1,2,3,4,0,3])
        >>> values = np.ones(6)
        >>> g = sp.coo_matrix((values, (row_indices, col_indices)))
        >>> g = g + sp.eye(g.shape[0])
        >>> g = g.eliminate_zeros()
        >>> indices = torch.from_numpy(np.asarray([g.row, g.col]))
        >>> values = torch.from_numpy(g.data)
        >>> g = torch.sparse_coo_tensor(indices, values, g.shape)
        >>> feat = torch.ones(6, 10)
        >>> gatconv = GATLayer(10, 2, 3)
        >>> res = gatconv(feat, g)
        >>> res
        tensor([[[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]],
                [[ 0.0268,  1.0783],
                [ 0.5041, -1.3025],
                [ 0.6568,  0.7048]],
                [[-0.2688,  1.0543],
                [-0.0315, -0.9016],
                [ 0.3943,  0.5347]],
                [[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)

        >>> row_indices = np.array([0, 1, 0, 0, 1])
        >>> col_indices = np.array([0, 1, 2, 3, 2])
        >>> values = np.ones(5)
        >>> g = sp.coo_matrix((values, (row_indices, col_indices)))
        >>> g = torch.sparse_coo_tensor(torch.from_numpy(np.asarray([g.row, g.col])), torch.from_numpy(g.data), g.shape)
        >>> u_feat = torch.tensor(np.random.rand(2, 5).astype(np.float32))
        >>> v_feat = torch.tensor(np.random.rand(4, 10).astype(np.float32))
        >>> gatconv = GATLayer((5,10), 2, 3)
        >>> res = gatconv((u_feat, v_feat), g)
        >>> res
        tensor([[[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]],
                [[ 0.0268,  1.0783],
                [ 0.5041, -1.3025],
                [ 0.6568,  0.7048]],
                [[-0.2688,  1.0543],
                [-0.0315, -0.9016],
                [ 0.3943,  0.5347]],
                [[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.in_feats = in_feats
        self.allow_zero_in_degree = allow_zero_in_degree

        # Handling separate transformations for bipartite graphs
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(in_feats[0], out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(in_feats[1], out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.softmax = nn.Softmax(dim=1)

        self.residual = residual
        self.activation = activation

        if residual:
            if isinstance(in_feats, tuple):
                if in_feats[1] != out_feats * num_heads:
                    self.res_fc = nn.Linear(in_feats[1], out_feats * num_heads,
                                            bias=bias)
                else:
                    self.res_fc = nn.Identity()
            else:
                if in_feats != out_feats * num_heads:
                    self.res_fc = nn.Linear(in_feats, out_feats * num_heads, bias=bias)
                else:
                    self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias and not (isinstance(self.res_fc, nn.Linear) and residual):
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_feats))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, g, h, edge_weight=None, get_attention=False):
        r"""
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        h : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        g : torch.sparse_coo_tensor
            The graph.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        ValueError
            If there are 0-in-degree nodes in the input graph, it will raise ValueError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        # if not self.allow_zero_in_degree and (g.sum(dim=1) == 0).any():
        if not self.allow_zero_in_degree and (torch.sparse.sum(g) == 0).any():
            raise ValueError(
                "There are 0-in-degree nodes in the graph, output for those nodes will be invalid. Adding self-loops will resolve the issue.")

        if isinstance(h, tuple):
            h_src = self.feat_drop(h[0])
            h_dst = self.feat_drop(h[1])
            if not hasattr(self, 'fc_src'):
                Wh_src = self.fc(h_src).view(-1, self.num_heads, self.out_feats)
                Wh_dst = self.fc(h_dst).view(-1, self.num_heads, self.out_feats)
            else:
                Wh_src = self.fc_src(h_src).view(-1, self.num_heads, self.out_feats)
                Wh_dst = self.fc_dst(h_dst).view(-1, self.num_heads, self.out_feats)
        else:
            h = self.feat_drop(h)
            Wh = self.fc(h).view(-1, self.num_heads, self.out_feats)
            Wh_src = Wh_dst = Wh

        # Compute attention scores
        e_l = (Wh_src * self.attn_l).sum(dim=-1)  # Shape (N, 1, num_heads)
        e_r = (Wh_dst * self.attn_r).sum(dim=-1)  # Shape (1, N, num_heads)

        e = e_l[g._indices()[0, :].t(), :] + e_r[g._indices()[1, :].t(), :]
        e = self.leaky_relu(e)

        # Compute edge softmax, we use torch.sparse.softmax here
        # construct a sparse tensor for edge attention scores
        # shape （N, N, num_heads)
        shape = g.shape  # shape (N, N)
        # add num_heads dimension to the shape
        shape = shape + (self.num_heads,)
        attention = torch.sparse_coo_tensor(g._indices(), e, shape)
        attention = torch.sparse.softmax(attention, dim=0)

        attention = attention.coalesce().values()

        attention = self.attn_drop(attention)

        if edge_weight is not None:
            edge_weight = edge_weight.unsqueeze(1).expand(-1, self.num_heads)
            # edge_weight = torch.sparse_coo_tensor(g._indices(), edge_weight, shape)
            attention = attention * edge_weight

        h_prime = []
        for i in range(self.num_heads):
            head_values = attention[:, i]
            g = g.coalesce()
            attention_i = torch.sparse_coo_tensor(g.indices(), head_values,
                                                  g.shape)
            # h_prime.append(torch.spmm(attention_i, Wh_dst[:, i]))
            h_prime.append(torch.spmm(attention_i.t(), Wh_src[:, i]))
        h_prime = torch.stack(h_prime, dim=1)

        if self.residual:
            if isinstance(h, tuple):
                resval = self.res_fc(h[1]).view(-1, self.num_heads, self.out_feats)
            else:
                resval = self.res_fc(h).view(-1, self.num_heads, self.out_feats)
            h_prime += resval

        if self.bias is not None:
            h_prime += self.bias.view(1, -1, self.out_feats)

        if self.activation:
            h_prime = self.activation(h_prime)

        if get_attention:
            return h_prime, attention
        else:
            return h_prime


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, z, attention=False):
        w = self.project(z)  # (M, 1)
        if attention:
            beta_copy = w
        w = w.mean(0)
        beta = self.softmax(w)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        if attention:
            return (beta * z).sum(1), beta_copy

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    gs : list[torch.SparseTensor]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        self.is_conv = True

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h, gat_attention=False, semantic_attention=False):
        semantic_embeddings = []
        if not gat_attention:
            for i, g in enumerate(gs):
                if isinstance(h, tuple) or isinstance(h, list):
                    semantic_embeddings.append(
                        self.gat_layers[i](g, h[i]).flatten(1))
                else:
                    semantic_embeddings.append(
                        self.gat_layers[i](g, h).flatten(1))  # (N, D * K)
            semantic_embeddings = torch.stack(semantic_embeddings,
                                              dim=1)  # (N, M, D * K)
        else:
            gat_attention_weights = []
            for i, g in enumerate(gs):
                if isinstance(h, tuple) or isinstance(h, list):
                    semantic_embedding, attention = self.gat_layers[i](g, h[i],
                                                                       get_attention=True)
                else:
                    semantic_embedding, attention = self.gat_layers[i](g, h,
                                                                       get_attention=True)
                semantic_embeddings.append(semantic_embedding.flatten(1))
                gat_attention_weights.append(attention)
            semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        if semantic_attention:
            output, beta = self.semantic_attention(semantic_embeddings, attention=True)
            if not gat_attention:
                return output, beta
            return output, beta, gat_attention_weights

        if not gat_attention:
            return self.semantic_attention(semantic_embeddings)  # (N, D * K)
        return self.semantic_attention(semantic_embeddings), gat_attention_weights


class HANEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_heads, dropout, num_meta_paths=3):
        super(HANEncoder, self).__init__()

        self.layers = nn.ModuleList()
        first_layer_out = hidden_size if len(num_heads) > 1 else out_size
        self.layers.append(HANLayer(num_meta_paths=num_meta_paths, in_size=in_size,
                                    out_size=first_layer_out, layer_num_heads=num_heads[0],
                                    dropout=dropout))
        if len(num_heads) > 2:
            for l in range(1, len(num_heads) - 1):
                self.layers.append(HANLayer(num_meta_paths=num_meta_paths, in_size=hidden_size * num_heads[l - 1],
                                            out_size=hidden_size, layer_num_heads=num_heads[l],
                                            dropout=dropout))
        if len(num_heads) > 1:
            self.layers.append(HANLayer(num_meta_paths=num_meta_paths, in_size=hidden_size * num_heads[-2],
                                        out_size=out_size, layer_num_heads=num_heads[-1],
                                        dropout=dropout))

    def forward(self, gs, h):
        for gnn in self.layers:
            h = gnn(gs, h)
        return h


class embedding_layer(nn.Module):
    def __init__(self, in_size):
        super(embedding_layer, self).__init__()
        self.embed = nn.Linear(in_size, 1)

    def forward(self, h1, h2):
        h = torch.cat([h1, h2], dim=1)
        return self.embed(h).squeeze()


class HANLinkPredictor(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_heads1, num_heads2, dropout, encoder1_num_meta_paths=3,
                 encoder2_num_meta_paths=3):
        super(HANLinkPredictor, self).__init__()
        self.encoder1 = HANEncoder(in_size, hidden_size, out_size, num_heads1, dropout, encoder1_num_meta_paths)
        self.encoder2 = HANEncoder(in_size, hidden_size, out_size, num_heads2, dropout, encoder2_num_meta_paths)
        self.embed_layer = nn.Linear(out_size * num_heads1[-1] + out_size * num_heads2[-1], 1)
        self.embed_layer_hyper = out_size * num_heads1[-1] + out_size * num_heads2[-1]

    def forward(self, gs1, gs2, h):
        h1 = self.encoder1(gs1, h)
        h2 = self.encoder2(gs2, h)
        return h1, h2

    def embed_link(self, h1, h2):
        h = torch.cat([h1, h2], dim=1)
        return self.embed_layer(h).squeeze()
