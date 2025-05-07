import os.path

import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedgvd.SimGC_transductive import data_gc, root, args
from openfgl.flcore.fedgvd.fedgvd_config import config

class Feddgc1Client(BaseClient):
    """
    FedProtoClient is a client implementation for the Federated Prototype Learning (FedProto) framework.
    This client handles the local training of models, computes class-specific prototypes, and interacts
    with the server to contribute to the global model updates.

    Attributes:
        local_prototype (dict): A dictionary storing the local prototypes for each class after training.
    """


    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedProtoClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(Feddgc1Client, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.message_pool["feat_syn"] = {}
        self.message_pool["edge_index_syn"] = {}
        self.message_pool["label_syn"] = {}
        self.message_pool["syn_logits"] = []
        self.local_data_len = self.task.data.num_nodes

        self.feat_syn = {}
        self.edge_index_syn = {}
        self.label_syn = {}
        self.syn_logits = []

    def execute(self):
        """
        Executes the local training process. This method sets a custom loss function that incorporates
        the prototype-based regularization term, performs local training, and then updates the local
        prototypes for each class.
        """
        if self.message_pool["round"] == 0:
            if os.path.exists( f'{root}/saved_ours/feat_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{self.client_id}.pt') and os.path.exists(f'{root}/saved_ours/adj_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{self.client_id}.pt') and os.path.exists(f'{root}/saved_ours/label_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{self.client_id}.pt'):
                feat_syn = torch.load(
                    f'{root}/saved_ours/feat_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{self.client_id}.pt').detach().to(self.device)
                adj_syn = torch.load(
                    f'{root}/saved_ours/adj_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{self.client_id}.pt').detach().to(self.device)
                labels_syn = torch.load(
                    f'{root}/saved_ours/label_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{self.client_id}.pt').detach().to(self.device)
            else:
                feat_syn,adj_syn,labels_syn = data_gc(self.task.processed_data,self.client_id)

            random_matrix  = torch.rand(adj_syn.shape).to(self.device)  
            sampled_edges = (random_matrix <= adj_syn).float()

            rows, cols = torch.where(sampled_edges != 0)

            edge_index_syn = torch.stack([rows, cols], dim=0)
            self.message_pool["feat_syn"][self.client_id] = feat_syn
            self.message_pool["edge_index_syn"][self.client_id] = edge_index_syn
            self.message_pool["label_syn"][self.client_id] = labels_syn

            self.feat_syn = feat_syn
            self.edge_index_syn = edge_index_syn
            self.label_syn = labels_syn

        elif self.message_pool["round"] == 1:
            global_feat_syn, global_label_syn, global_edge_index_syn = self.cat_syn_data()

            self.task.processed_data['data']['x'] = torch.cat([
                self.task.processed_data['data']['x'],
                self.feat_syn 
            ])
            self.task.processed_data['data']['y'] = torch.cat([
                self.task.processed_data['data']['y'],
                self.label_syn  
            ])

            
            self.task.processed_data['data']['edge_index'] = torch.cat([
                self.task.processed_data['data']['edge_index'], 
                self.edge_index_syn  
            ], dim=1)
            self.task.processed_data['train_mask'] = torch.cat([self.task.processed_data['train_mask'], torch.ones(self.feat_syn.shape[0], dtype=torch.bool).to(self.device)], dim=0)
            self.task.processed_data['val_mask'] = torch.cat([self.task.processed_data['val_mask'],
                                                                torch.zeros(self.feat_syn.shape[0],
                                                                           dtype=torch.bool).to(self.device)], dim=0)
            self.task.processed_data['test_mask'] = torch.cat([self.task.processed_data['test_mask'],
                                                                torch.zeros(self.feat_syn.shape[0],
                                                                           dtype=torch.bool).to(self.device)], dim=0)
            self.task.train_mask = self.task.processed_data['train_mask']
            self.task.val_mask = self.task.processed_data['val_mask']
            self.task.test_mask = self.task.processed_data['test_mask']

            self.task.loss_fn = self.get_custom_loss_fn()
            self.task.train()
            self.update_syn_logits()
        else:
            self.task.loss_fn = self.get_custom_loss_fn()
            self.task.train()
            self.update_syn_logits()


    def get_custom_loss_fn(self):
        """

        Returns:
            custom_loss_fn (function): A custom loss function.
        """
        def custom_loss_fn(embedding, logits, label, mask):
            if self.message_pool["round"] == 1:
                return self.task.default_loss_fn(logits[:self.local_data_len][mask[:self.local_data_len]], label[:self.local_data_len][mask[:self.local_data_len]]) + config['gama']*self.task.default_loss_fn(logits[self.local_data_len:][mask[self.local_data_len:]], label[self.local_data_len:][mask[self.local_data_len:]])
            else:
                loss_kd = 0
                input = logits[self.local_data_len:]
                target = self.message_pool[f"client_{self.client_id}"]["syn_logits"].to(self.device)
                loss_kd += nn.MSELoss()(input, target)
                return self.task.default_loss_fn(logits[:self.local_data_len][mask[:self.local_data_len]], label[:self.local_data_len][mask[:self.local_data_len]]) + config['gama']*self.task.default_loss_fn(logits[self.local_data_len:][mask[self.local_data_len:]], label[self.local_data_len:][mask[self.local_data_len:]]) + config['lambda'] * loss_kd
        return custom_loss_fn


    def update_syn_logits(self):
        """
        Updates the local prototypes for each class after local training. The prototypes are calculated
        as the mean of the embeddings of the samples belonging to each class.
        """

        with torch.no_grad():
            embedding = self.task.evaluate(mute=True)["embedding"]
            shift = self.local_data_len
            # for i in range(1,self.client_id):
            #     shift += len(self.message_pool["feat_syn"].get(i-1))
            # update_logits = embedding[shift:shift+len(self.message_pool["feat_syn"].get(self.client_id))]
            update_logits = embedding[shift: shift + len(self.feat_syn)]
            # print(len(self.feat_syn))
            if self.message_pool["round"] == 1:
                self.message_pool["syn_logits"].append(update_logits)
            else:
                self.message_pool["syn_logits"][self.client_id] = update_logits

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "feat_syn": self.feat_syn,
            "edge_index_syn": self.edge_index_syn,
            "label_syn": self.label_syn,
            "syn_logits": self.syn_logits
        }

    def cat_syn_data(self):
        feat_syn = list(self.message_pool["feat_syn"].values())
        global_feat_syn = torch.cat(feat_syn, dim=0)
        label_syn = list(self.message_pool["label_syn"].values())
        global_label_syn = torch.cat(label_syn, dim=0)
        edge_index_syn = list(self.message_pool["edge_index_syn"].values())

        node_offset = 0
        all_edge_index_syn = []
        for i in range(len(edge_index_syn)):
            
            edge_index = edge_index_syn[i]

            edge_index_offset = edge_index + node_offset

            all_edge_index_syn.append(edge_index_offset)

            
            node_offset += feat_syn[i].shape[0]  # 或 clients_label[i].shape[0]

        global_edge_index_syn = torch.cat(all_edge_index_syn, dim=1)
        return global_feat_syn, global_label_syn, global_edge_index_syn
