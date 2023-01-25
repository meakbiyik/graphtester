from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster
import shutil
from pathlib import Path

from model import GNN
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv  # noqa: F401
from train import train

import graphtester as gt


def main(args, cluster=None):
    print(args)

    SCRIPT_PATH = Path(__file__).parent
    dataset = TUDataset(
        root=SCRIPT_PATH / "data" / args.dataset / args.features,
        name="MUTAG",
        pre_transform=gt.pretransform(
            features=[args.features],
            feature_names="x",
        ),

    )

    node_feature_count = dataset[0].x.shape[1]
    num_classes = dataset.num_classes

    model = GNN(node_feature_count, num_classes, args.hidden_units, args.dropout)
    train(dataset, model, args.batch_size)
    print('Done')



if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--dropout', type=float, default=0.5, tunable=True, options=[0.5, 0.0])
    parser.opt_list('--batch_size', type=int, default=32, tunable=True, options=[32, 128])
    parser.opt_list('--hidden_units', type=int, default=64, tunable=True, options=[16, 32])
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.opt_list('--features', type=str, default='Closeness centrality', tunable=True, options=['Closeness centrality', 'Betweenness centrality', 'Harmonic centrality', 'Local transitivity'])

    args = parser.parse_args()

    if args.slurm:
        print('Launching SLURM jobs')
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path='slurm_log/',
            python_cmd='python'
        )
        cluster.job_time = '24:00:00'
        
        cluster.memory_mb_per_node = '6G'
        job_name = "AutoGNN"
        cluster.per_experiment_nb_gpus = 0
        cluster.per_experiment_nb_cpus = 1
        cluster.optimize_parallel_cluster_cpu(main, nb_trials=None, job_name=job_name, job_display_name=args.dataset)
    else:
        main(args)

    print('Finished', flush=True)
