import torch
from openfgl.flcore.base import BaseServer
from ..feddgc.contrastive import get_coordinated_data
from torch_geometric.data import Data
from torch import nn


class Feddgc1Server(BaseServer):

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(Feddgc1Server, self).__init__(args, global_data, data_dir, message_pool, device)


    def execute(self):
        if self.message_pool["round"] == 0:
            self.global_data = self._aggregate_subgraphs()
        self._train_global_model(self.global_data)
        self._calculate_client_logits(self.global_data)

    def send_message(self):
        pass

    def _aggregate_subgraphs(self):
        datalist = []
        for client_id in self.message_pool["sampled_clients"]:
            client_data = self.message_pool[f"client_{client_id}"]

            x = torch.tensor(client_data["feat_syn"], dtype=torch.float) \
                if not isinstance(client_data["feat_syn"], torch.Tensor) \
                else client_data["feat_syn"].float()

            edge_index = torch.tensor(client_data["edge_index_syn"], dtype=torch.long) \
                if not isinstance(client_data["edge_index_syn"], torch.Tensor) \
                else client_data["edge_index_syn"].long()

            if edge_index.dim() == 1:
                edge_index = edge_index.view(2, -1)
            elif edge_index.size(0) != 2:
                edge_index = edge_index.t().contiguous()

            data = Data(x=x, edge_index=edge_index)

            if "label_syn" in client_data:
                y = torch.tensor([client_data["label_syn"]], dtype=torch.long) \
                    if not isinstance(client_data["label_syn"], torch.Tensor) \
                    else client_data["label_syn"]
                data.y = y

            datalist.append(data)

        return get_coordinated_data(
            datalist=datalist,
            cross_link=1,
            dynamic_edge='none',
            dynamic_prune=0.5,
            cross_link_ablation=False,
            device=self.device
        )

    def _train_global_model(self, global_data):
        self.task.model.train()
        global_data = global_data.to(self.device)

        for _ in range(10):
            self.task.optim.zero_grad()

            embeddings, logits = self.task.model(global_data)

            loss = self._calculate_loss(global_data, logits)

            loss.backward()
            self.task.optim.step()

    def _calculate_loss(self, data, logits):
        return nn.CrossEntropyLoss()(logits[:-30], data.y)

    def _calculate_client_logits(self, global_data):
        self.task.model.eval()
        with torch.no_grad():
            _, all_logits = self.task.model(global_data)

            client_logits = []
            ptr = 0
            for cid in self.message_pool["sampled_clients"]:
                num_nodes = self.message_pool[f"client_{cid}"]["feat_syn"].shape[0]
                client_logits.append(all_logits[ptr:ptr + num_nodes])
                ptr += num_nodes
                
            for cid, logits in zip(self.message_pool["sampled_clients"], client_logits):
                self.message_pool[f"client_{cid}"]["syn_logits"] = logits.cpu()

