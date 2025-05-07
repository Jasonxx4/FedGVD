import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "dataset"


args.dataset = ["Computers"]
args.simulation_mode = "subgraph_fl_louvain"
args.num_clients = 10


# if True:
#     args.fl_algorithm = "fedavg"
#     args.model = ["gcn"]
# else:
args.fl_algorithm = "fedgvd"
# args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"] # choose multiple gnn models for model heterogeneity setting.
args.model = ["gcn","gat", "sgc","graphsage"] # choose multiple gnn models for model heterogeneity setting.
# args.model = ["gcn","gat"] # choose multiple gnn models for model heterogeneity setting.
# args.model = ["gcn"] # choose multiple gnn models for model heterogeneity setting.

args.metrics = ["accuracy"]


trainer = FGLTrainer(args)

trainer.train()
