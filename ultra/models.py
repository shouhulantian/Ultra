import torch
from torch import nn
from . import tasks, layers
from ultra.base_nbfnet import BaseNBFNet
import torch_geometric
import numpy as np
import pickle
from ultra.datasets import ICEWS14Ind
#from ultra.util import tranform_reccurrency2ultra
from ultra.tasks import build_relation_graph

class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg, rule_model_cfg=None):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()

        self.relation_model = RelNBFNet(**rel_model_cfg)
        self.entity_model = EntityNBFNet(**entity_model_cfg)
        if rule_model_cfg is not None:
            self.rule_model = Reccurency(**rule_model_cfg)
        self.window_size = rel_model_cfg['window_size']
        if self.window_size>0:
            feature_dim = self.entity_model.dims[0]*4
            self.mlp = nn.Sequential()
            mlp = []
            for i in range(1):
                mlp.append(nn.Linear(feature_dim, feature_dim))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(feature_dim, 1))
            self.mlp = nn.Sequential(*mlp)
        
    def forward(self, data, batch):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        query_times = batch[:, 0, 3]
        #score_rule, alpha = self.rule_model(data,batch)
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        score, output = self.entity_model(data, relation_representations, batch)
        if self.window_size > 0:
            entity_graph_t, relation_graph_t = self.generate_graph_t(data, query_times, self.window_size)
            #relation_representations_t = self.relation_model(entity_graph_t[i].relation_graph, query=query_rels)
            output_t = []
            score_t = []
            for i in range(len(entity_graph_t)):
                score_t_ind, output_t_ind = self.entity_model(entity_graph_t[i], relation_representations[i,:].unsqueeze(0), batch[i,:].unsqueeze(0))
                output_t.append(output_t_ind)
                score_t.append(score_t_ind)
            output_t = torch.stack(output_t).squeeze(dim=1)
            score_t = torch.stack(score_t).squeeze(dim=1)
            output = torch.cat([output, output_t], dim=-1)
            score = self.mlp(output).squeeze(-1)
        #relation_representations_t = self.relation_model(data.relation_graph, query_rels, query_times)
        # score_rule,alpha = self.rule_model(data,batch)
        # if alpha!=0:
        #     score = score_rule*alpha + score * (1-alpha)
        
        return score

    def generate_graph_t(self, data, times, window_size=3):
        time_start = times - window_size
        time_end = times + window_size

        relation_graph_t = []
        entity_graph_t = []
        for i in range(times.shape[0]):
            index = torch.ge(data.time_type, time_start[i]) & torch.le(data.time_type, time_end[i])
            edge_subset = data.edge_index[:,index]
            edge_type_subset=data.edge_type[index]
            time_type_subset = data.time_type[index]
            graph_t = torch_geometric.data.Data(edge_index=edge_subset, edge_type= edge_type_subset,
                                                                                    num_nodes=data.num_nodes,
                                                                                    num_relations=data.num_relations,time_type=time_type_subset,
                                                                                    num_time=data.num_time)
            graph_t.relation_graph = build_relation_graph(graph_t)
            entity_graph_t.append(graph_t)
            relation_graph_t.append(graph_t.relation_graph)

        return entity_graph_t, relation_graph_t
        # #train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
        #                   target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
        #                   num_relations=num_relations * 2, num_time=num_time, time_type=train_ttypes, target_time_type=train_target_ttypes)

# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, time_graph = 'null', window_size=0 , **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )
        self.time_graph = time_graph
        self.window_size = window_size

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            if time_graph != 'null':
                feature_dim = feature_dim + self.dims[0]
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

        if 'ind' in self.time_graph:
            self.layers_t = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layers_t.append(
                    layers.GeneralizedRelationalConv(
                        self.dims[i], self.dims[i + 1], num_relation,
                        self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                        self.activation, dependent=False)
                )
    
    def bellmanford(self, data, h_index, nbf_layers='static', separate_grad=False):
        try:
            batch_size = len(h_index)
        except:
            batch_size = 1
        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(batch_size, self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        if nbf_layers == 'static':
            for layer in self.layers:
                # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
                hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
                if self.short_cut and hidden.shape == layer_input.shape:
                    # residual connection here
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                edge_weights.append(edge_weight)
                layer_input = hidden
        else:
            for layer in self.layers_t:
                # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
                hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
                if self.short_cut and hidden.shape == layer_input.shape:
                    # residual connection here
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                edge_weights.append(edge_weight)
                layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query, relation_graph_t=None):

        # message passing and updated node representations (that are in fact relations)
        if 'r_s_t_concat' in self.time_graph:
            output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
            output_t = []
            if 'ind' in self.time_graph:
                for i in range(len(relation_graph_t)):
                    output_t.append(self.bellmanford(relation_graph_t[i].relation_graph,h_index=query[i],nbf_layers='ind')["node_feature"])
            else:
                for i in range(len(relation_graph_t)):
                    output_t.append(self.bellmanford(relation_graph_t[i].relation_graph,h_index=query[i])["node_feature"])
            output_t = torch.stack(output_t).squeeze(dim=1)
            output = torch.cat([output, output_t],dim=-1)
        elif 'r_t' in self.time_graph:
            output_t = []
            if 'ind' in self.time_graph:
                for i in range(len(relation_graph_t)):
                    output_t.append(self.bellmanford(relation_graph_t[i].relation_graph,h_index=query[i],nbf_layers='ind')["node_feature"])
            else:
                for i in range(len(relation_graph_t)):
                    output_t.append(self.bellmanford(relation_graph_t[i].relation_graph,h_index=query[i])["node_feature"])
            output = torch.stack(output_t).squeeze(dim=1)
        else:
            output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        return output
    

class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, use_time='null', num_relation=1, num_time=365, remove_edge='default', project_times=True, boundary='default', time_dependent=False, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        self.use_time = use_time
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True, time_dependent=time_dependent, project_times=project_times, num_time=num_time)
            )
        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        if 'concat' in self.use_time:
            feature_dim = feature_dim + self.dims[0]
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)
        self.remove_edge = remove_edge
        self.project_times = project_times
        self.boundary = boundary
        self.num_time = num_time
        self.time_dependent = time_dependent

        if 'nbf' in self.use_time:
            if self.time_dependent or not self.project_times:
                # relation embeddings as an independent embedding matrix per each layer
                self.time_projection = nn.Embedding(self.num_time, input_dim)
            else:
                # will be initialized after the pass over relation graph
                self.time_projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )
    
    def bellmanford(self, data, h_index, r_index, time_index=None, separate_grad=False):
        try:
            batch_size = len(h_index)
        except:
            batch_size = 1

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        if self.boundary == 'default':
            boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        elif 'time' in self.boundary:
            boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
            #index = time_index.unsqueeze(-1).expand_as(query)
            if self.time_dependent or not self.project_times:
                query = self.time_query(time_index)
            else:
                query = self.time_query[torch.arange(batch_size, device=r_index.device), time_index]
            boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight=edge_weight, time_type=data.time_type)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end.item(), device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cos = torch.cos(freqs)  # real part
        freqs_sin = torch.sin(freqs)  # imaginary part
        return freqs_cos, freqs_sin

    def forward(self, data, relation_representations, batch, entity_graph_t=None):
        if batch.shape[2] == 3:
            h_index, t_index, r_index = batch.unbind(-1)
        else:
            h_index, t_index, r_index, time_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations
        freqs_cos, freqs_sin = self.precompute_freqs_cis(self.dims[0], data.num_time)
        #self.time_query = torch.cat([freqs_cos, freqs_sin], dim=-1).expand(batch.shape[0], -1, -1).to(batch.device)
        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # if 'nbf' in self.use_time:
        #     #freqs_cos, freqs_sin = freqs_cos[time_index], freqs_sin[time_index]
        #     if self.project_times:
        #         for layer in self.layers:
        #             layer.time = torch.cat([freqs_cos,freqs_sin],dim=-1).expand(batch.shape[0], -1, -1).to(batch.device)
        #         self.time_query = self.layers[0].time_projection(self.layers[0].time)
        #     else:
        #         for layer in self.layers:
        #             layer.num_time = int(data.num_time)
        #             layer.time = torch.nn.Embedding(layer.num_time, layer.input_dim).to(time_index.device)
        #         self.time_query = layer.time

        if 'nbf' in self.use_time:
            #freqs_cos, freqs_sin = freqs_cos[time_index], freqs_sin[time_index]
            if self.time_dependent or not self.project_times:
                self.time_query = self.time_projection
            else:
                time_query = torch.cat([freqs_cos,freqs_sin],dim=-1).expand(batch.shape[0], -1, -1).to(batch.device)
                self.time_query = self.time_projection(time_query)
                for layer in self.layers:
                    layer.time = self.time_query

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            if self.remove_edge == 'time':
                data = self.remove_easy_edges(data, h_index, t_index, r_index, time_index=time_index)
            elif self.remove_edge == 'default':
                data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # # message passing and updated node representations
        # output = self.bellmanford(data, h_index[:, 0], r_index[:, 0],time_index=time_index[:,0])  # (num_nodes, batch_size, feature_dim）

        # message passing and updated node representations
        if 'r_s_t_concat' in self.use_time:
            output = self.bellmanford(data, h_index[:, 0], r_index[:, 0],time_index=time_index[:,0])  # (batch_size, num_nodes, hidden_dim）
            output_t = []
            for i in range(len(entity_graph_t)):
                output_t.append(self.bellmanford(entity_graph_t[i], h_index[i, 0], r_index[i, 0],time_index=time_index[i,0])["node_feature"])
            output_t = torch.stack(output_t).squeeze(dim=1)
            output = torch.cat([output, output_t],dim=-1)
        elif 'r_t' in self.use_time:
            output_t = []
            for i in range(len(entity_graph_t)):
                output_t.append(self.bellmanford(entity_graph_t[i], h_index[i, 0], r_index[i, 0],time_index=time_index[i,0])["node_feature"])
            output = torch.stack(output_t).squeeze(dim=1)
        else:
            output = self.bellmanford(data, h_index[:, 0], r_index[:, 0],
                                      time_index=time_index[:, 0])["node_feature"]  # (num_nodes, batch_size, feature_dim）

        feature = output
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        if 'complex' in self.use_time:
            freqs_cos, freqs_sin = self.precompute_freqs_cis(self.dims[0], data.num_time)
            freqs_cos = freqs_cos.to(time_index.device)
            freqs_sin = freqs_sin.to(time_index.device)
            freqs_cos, freqs_sin = freqs_cos[time_index], freqs_sin[time_index]

            feature_h, feature_r = feature[:,:, :self.dims[0]], feature[:,:, self.dims[0]:]
            feature_h_r = feature_h[:,:, :self.dims[0]//2] * freqs_cos - feature_h[:,:, self.dims[0]//2:] * freqs_sin
            feature_h_i = feature_h[:,:, :self.dims[0]//2] * freqs_sin + feature_h[:,:, self.dims[0]//2:] * freqs_cos
            feature_r_r = feature_r[:,:, :self.dims[0]//2] * freqs_cos - feature_r[:,:, self.dims[0]//2:] * freqs_sin
            feature_r_i = feature_r[:,:, :self.dims[0]//2] * freqs_sin + feature_r[:,:, self.dims[0]//2:] * freqs_cos
            feature = torch.cat([feature_h_r, feature_h_i,feature_r_r,feature_r_i],dim=-1)
        elif 'concat' in self.use_time:
            freqs_cos, freqs_sin = self.precompute_freqs_cis(self.dims[0], data.num_time)
            freqs_cos = freqs_cos.to(time_index.device)
            freqs_sin = freqs_sin.to(time_index.device)
            freqs_cos, freqs_sin = freqs_cos[time_index], freqs_sin[time_index]
            feature = torch.cat([feature,freqs_cos,freqs_sin],dim=-1)
        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape), output

class Reccurency(nn.Module):
    def __init__(self, alpha=0,  **kwargs):
        # kept that because super Ultra sounds cool
        super(Reccurency, self).__init__()
        self.alpha = alpha

    def forward(self, data, batch):
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        if batch.shape[2] == 3:
            h_index, t_index, r_index = batch.unbind(-1)
        else:
            h_index, t_index, r_index, time_index = batch.unbind(-1)

        train_edges = data.edge_index
        train_rels = data.edge_type

        query_distributions = []

        # Iterate through each query
        for i in range(batch.size(0)):
            query_start = h_index[i,0]  # Starting node of the query
            query_label = r_index[i,0]  # Edge label of the query

            # Find edges that match the query start and label
            matching_edges = (train_edges[0] == query_start) & (train_rels == query_label)

            #matching_edges = (train_edges[0] == query_start)

            # Get the corresponding ending nodes
            ending_nodes = train_edges[1, matching_edges]

            # Calculate the frequency distribution of ending nodes
            distribution = torch.bincount(ending_nodes, minlength=data.num_nodes)

            # Store the distribution for this query
            query_distributions.append(distribution)

        # Convert to tensor for better visualization
        query_distributions = torch.stack(query_distributions)
        if torch.cuda.is_available():
            query_distributions = query_distributions.to(batch.device)
        freq_res = torch.softmax(query_distributions.float(), dim=1)

        return freq_res, self.alpha

# class Reccurency(nn.Module):
#
#     def __init__(self, score_path, num_relation, alpha=0,  **kwargs):
#         # kept that because super Ultra sounds cool
#         super(Reccurency, self).__init__()
#
#         self.score_path = score_path
#         self.alpha = alpha
#         self.num_relation = num_relation
#         self.ts, self.triples, self.scores = self.tranform_reccurrency2ultra(self.score_path,self.num_relation//2)
#         #self.ts, self.triples, self.scores = self.restructure_pickle_file(pickle.load(open(self.score_path,'rb')),self.num_relation)
#
#     def forward(self, data, batch):
#         # batch shape: (bs, 1+num_negs, 3)
#         # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
#         if batch[0,0,3].item() not in self.ts:
#             score = 0
#         else:
#             score = self.fetch_all_scores(batch,self.ts,self.triples,self.scores)*self.alpha
#
#         return score, self.alpha
#
#     def fetch_all_scores(self,main_tensor, timestamps, triples_list, scores_list):
#         batch_size, num_quadruples, _ = main_tensor.size()
#         all_scores = torch.zeros((batch_size, num_quadruples)).to(main_tensor.device)
#
#         for batch_idx in range(batch_size):
#             relation = main_tensor[batch_idx, 0, 2]
#             head = main_tensor[batch_idx, 0, 0]
#             timestamp = main_tensor[batch_idx,0,3]
#             #
#             # for quadruple_idx in range(num_quadruples):
#             #     quadruple = main_tensor[batch_idx, quadruple_idx].tolist()
#             #     head, tail, relation, timestamp = quadruple
#
#             if timestamp not in timestamps:
#                 all_scores[batch_idx,:] = 0
#                 continue
#
#                 # Find the index of the timestamp in the numpy array
#             timestamp_idx = timestamps.index(timestamp)
#             # Get the triples and scores for the corresponding timestamp
#             triples_tensor = triples_list[timestamp_idx]
#             scores_tensor = scores_list[timestamp_idx]
#
#             # Find the index of the triple in the triples tensor
#             triple_idx = -1
#             for i, triple in enumerate(triples_tensor):
#                 triple = triple.tolist()
#                 if triple[0] == head and triple[1] == relation:
#                     triple_idx = i
#                     break
#
#             if triple_idx == -1:
#                 all_scores[batch_idx, :] = 0
#             else:
#                 score = scores_tensor[triple_idx]
#                 all_scores[batch_idx, :] = torch.from_numpy(score).to(all_scores.device)
#
#         return all_scores
#
#     def restructure_pickle_file(self,pickle_file: dict, num_rels: int) -> list:
#         """
#         Restructure the pickle format to be able to use the functions in RE-GCN implementations.
#         The main idea is to use them as tensors so itspeeds up the computations
#         :param pickle_file:
#         :param num_rels:
#         :return:
#         """
#         pickle_file = pickle.load(open(self.score_path, 'rb'))
#         test_triples, final_scores, timesteps = [], [], []
#         for query, scores in pickle_file.items():
#             timestep = int(query.split('_')[-1])
#             timesteps.append(timestep)
#         timestepsuni = np.unique(timesteps)  # list with unique timestamps
#
#         timestepsdict_triples = {}  # dict to be filled with keys: timestep, values: list of all triples for that timestep
#         timestepsdict_scores = {}  # dict to be filled with keys: timestep, values: list of all scores for that timestep
#
#         for query, scores in pickle_file.items():
#             timestep = int(query.split('_')[-1])
#             triple = query.split('_')[:-1]
#             triple = np.array([int(elem.replace('xxx', '')) if 'xxx' in elem else elem for elem in triple],
#                               dtype='int32')
#             if query.startswith('xxx'):  # then it was subject prediction -
#                 triple = triple[np.argsort([2, 1, 0])]  # so we have to turn around the order
#                 triple[1] = triple[1] + num_rels  # and the relation id has to be original+num_rels to indicate it was
#                 # other way round
#
#             if timestep in timestepsdict_triples:
#                 timestepsdict_triples[timestep].append(torch.tensor(triple))
#                 timestepsdict_scores[timestep].append(torch.tensor(scores[0]))
#             else:
#                 timestepsdict_triples[timestep] = [torch.tensor(triple)]
#                 timestepsdict_scores[timestep] = [torch.tensor(scores[0])]
#
#         for t in np.sort(list(timestepsdict_triples.keys())):
#             test_triples.append(torch.stack(timestepsdict_triples[t]))
#             final_scores.append(torch.stack(timestepsdict_scores[t]))
#
#         return timestepsuni, test_triples, final_scores
#
#     def tranform_reccurrency2ultra(self,pickle_file, num_rels):
#         ts, triples, scores = self.restructure_pickle_file(pickle_file, num_rels)
#         datasets = ICEWS14Ind('~/git/ULTRA/kg-datasets/')
#         ent_vocab, rel_vocab, time_vocab = datasets.provide_vocab()
#         ts_convert, triples_convert, scores_convert = [], [], []
#         for t in ts:
#             t_new = time_vocab[str((t-1)*24)]
#             ts_convert.append(t_new)
#
#         for i in range(len(triples)):
#             snapshot = triples[i]
#             scoreshot = scores[i]
#             transformed_tensor = np.empty_like(snapshot)
#             transformed_scores = np.empty_like(scoreshot)
#
#             for i in range(snapshot.shape[0]):
#                 head_entity_idx = str(snapshot[i, 0].item())
#                 relation_idx = snapshot[i, 1].item()
#                 tail_entity_idx = str(snapshot[i, 2].item())
#
#                 head_entity_name = ent_vocab[head_entity_idx]
#                 if int(relation_idx) < num_rels:
#                     relation_name = rel_vocab[str(relation_idx)]
#                 else:
#                     relation_name = rel_vocab[str(relation_idx-num_rels)] + num_rels
#                 tail_entity_name = ent_vocab[tail_entity_idx]
#
#                 # # Transform to vocab indices
#                 # head_entity_vocab_idx = list(ent_vocab.keys())[list(ent_vocab.values()).index(head_entity_name)]
#                 # relation_vocab_idx = list(rel_vocab.keys())[list(rel_vocab.values()).index(relation_name)]
#                 # tail_entity_vocab_idx = list(ent_vocab.keys())[list(ent_vocab.values()).index(tail_entity_name)]
#
#                 transformed_tensor[i, 0] = head_entity_name
#                 transformed_tensor[i, 1] = relation_name
#                 transformed_tensor[i, 2] = tail_entity_name
#             triples_convert.append(transformed_tensor)
#
#             # Extract entity indices from the vocabulary
#             num_rows, num_cols = scoreshot.shape
#             # Sort the entity indices to reorder columns in score_list
#             for row in range(num_rows):
#                 transformed_scores[row, :] = scoreshot[row, [ent_vocab[str(i)] for i in range(num_cols)]]
#             scores_convert.append(transformed_scores)
#         output_log = {'ts_convert':ts_convert,
#                       'triples_convert':triples_convert,
#                       'scores_convert':scores_convert}
#         pickle.dump(output_log, open('ICEWS14Ind_converted.pkl','wb'), protocol=4)
#
#         return ts_convert, triples_convert, scores_convert

    

    


