import torch
from openfgl.flcore.base import BaseServer
from ..feddgc.contrastive import get_coordinated_data
from torch_geometric.data import Data
from torch import nn


class Feddgc1Server(BaseServer):

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(Feddgc1Server, self).__init__(args, global_data, data_dir, message_pool, device)


    def execute(self):
        """执行流程：聚合子图->训练全局模型->计算logits"""
        # 1. 聚合客户端合成数据
        if self.message_pool["round"] == 0:
            self.global_data = self._aggregate_subgraphs()
        # 2. 训练全局模型
        self._train_global_model(self.global_data)

        # 3. 计算客户端logits
        self._calculate_client_logits(self.global_data)

    def send_message(self):
        pass

    def _aggregate_subgraphs(self):
        datalist = []
        for client_id in self.message_pool["sampled_clients"]:
            client_data = self.message_pool[f"client_{client_id}"]

            # 转换节点特征和边索引为张量
            x = torch.tensor(client_data["feat_syn"], dtype=torch.float) \
                if not isinstance(client_data["feat_syn"], torch.Tensor) \
                else client_data["feat_syn"].float()

            edge_index = torch.tensor(client_data["edge_index_syn"], dtype=torch.long) \
                if not isinstance(client_data["edge_index_syn"], torch.Tensor) \
                else client_data["edge_index_syn"].long()

            # 确保边索引形状为 [2, num_edges]
            if edge_index.dim() == 1:
                edge_index = edge_index.view(2, -1)
            elif edge_index.size(0) != 2:
                edge_index = edge_index.t().contiguous()

            # 创建 Data 对象（必需字段 x 和 edge_index）
            data = Data(x=x, edge_index=edge_index)

            # 添加图标签（如果存在）
            if "label_syn" in client_data:
                y = torch.tensor([client_data["label_syn"]], dtype=torch.long) \
                    if not isinstance(client_data["label_syn"], torch.Tensor) \
                    else client_data["label_syn"]
                data.y = y

            # 添加到 datalist
            datalist.append(data)

        # 生成协调全局图
        return get_coordinated_data(
            datalist=datalist,
            cross_link=1,
            dynamic_edge='none',
            dynamic_prune=0.5,
            cross_link_ablation=False,
            device=self.device
        )

    def _train_global_model(self, global_data):
        """在协调全局图上训练模型"""
        self.task.model.train()
        global_data = global_data.to(self.device)

        for _ in range(10):
            self.task.optim.zero_grad()

            # 模型前向
            embeddings, logits = self.task.model(global_data)

            # 计算损失（示例使用协调器分类任务）
            loss = self._calculate_loss(global_data, logits)

            loss.backward()
            self.task.optim.step()

    def _calculate_loss(self, data, logits):
        """自定义损失函数（示例：协调器节点分类）"""
        # 假设协调器节点标签存储在data.y中
        return nn.CrossEntropyLoss()(logits[:-30], data.y)

    def _calculate_client_logits(self, global_data):
        """提取各客户端原始节点logits"""
        self.task.model.eval()
        with torch.no_grad():
            _, all_logits = self.task.model(global_data)

            # 按原始子图划分
            client_logits = []
            ptr = 0
            for cid in self.message_pool["sampled_clients"]:
                num_nodes = self.message_pool[f"client_{cid}"]["feat_syn"].shape[0]
                client_logits.append(all_logits[ptr:ptr + num_nodes])
                ptr += num_nodes

            # 保存到消息池
            for cid, logits in zip(self.message_pool["sampled_clients"], client_logits):
                self.message_pool[f"client_{cid}"]["syn_logits"] = logits.cpu()

