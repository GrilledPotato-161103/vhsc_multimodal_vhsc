import os
import time
import torch
import numpy as np
import torch.nn as nn
from DelaunayComponentAnalysis.dca.schemes import (
    DCALoggers,
    DelaunayGraphParams,
    ExperimentDirs,
    GeomCAParams,
    HDBSCANParams,
    REData,
)
from DelaunayComponentAnalysis.dca.DCA import DCA
import DelaunayComponentAnalysis.dca.visualization as DCA_visualization
from util_scripts.wandb_logger import WandbLogger
from util_scripts.train_callbacks import ModelSaverLoaderCallback



class DCAEvaluator():
    def __init__(self, model, dataset, data_module, opt):
        super(DCAEvaluator, self).__init__()

        self.model = model
        self.callback = ModelSaverLoaderCallback(opt.result_path, 'model', opt=opt)
        self.model = self.callback.load_cpkt(model, last=False)
        self.model.eval()
        self.dataset = dataset
        self.data_module = data_module
        self.opt = opt

        self.device = opt.device
        self.mcs = opt.minimum_cluster_size
        self.unique_modality_idxs = opt.unique_modality_idxs
        self.unique_modality_dims = opt.unique_modality_dims
        self.partial_modalities_idxs = opt.partial_modalities_idxs

        self.logger = WandbLogger(opt)

    def test_step(self, batch, batch_idx):
        if self.dataset in ['mosi', 'mosei']:
            batch_X, batch_Y, _ = batch[0], batch[1], batch[2]
            _, text, audio, vision = batch_X
            data = [text.to(self.device), audio.to(self.device), vision.to(self.device)]
        elif self.dataset in ['mmimdb', 'food101', 'hatememes']:
            batch_X, batch_Y = batch
            image, text = batch_X
            data = [text.to(self.device), image.to(self.device)]

        output_dict = {}
        # Forward pass through the encoder to get representations
        with torch.no_grad():
            batch_R_repr = self.model.encode(data, return_reps=True)
        output_dict[-1] = batch_R_repr

        # Drop modalities
        for k in range(len(data)):
            E_data = [None if k != j else data[k] for j in range(len(data))]
            with torch.no_grad():
                batch_E_repr = self.model.encode(E_data, return_reps=True)
            output_dict[k] = batch_E_repr
        return output_dict

    def evaluate(self, R, E, experiment_id):
        # initialize DCA params from ingredients
        data_config = REData(
            R=R, E=E, input_array_dir=os.path.join(experiment_id, "logs")
        )

        experiment_config = ExperimentDirs(
            experiment_dir=os.path.join(self.opt.result_path, 'dca'),
            experiment_id=experiment_id,
            precomputed_folder=os.path.join(experiment_id, "logs"),
        )

        graph_config = DelaunayGraphParams(
            unfiltered_edges_dir=os.path.join(experiment_id, "logs"),
            filtered_edges_dir=os.path.join(experiment_id, "logs"),
        )
        hdbscan_config = HDBSCANParams(
            min_cluster_size=self.mcs, clusterer_dir=os.path.join(experiment_id, "logs")
        )
        geomCA_config = GeomCAParams()
        exp_loggers = DCALoggers(experiment_config.logs_dir)

        # logging.config.dictConfig(exp_loggers.loggers)
        # logger = logging.getLogger("experiment_logger")

        DCA_algorithm = DCA(
            dirs=experiment_config,
            Delaunay_graph_params=graph_config,
            clustering_params=hdbscan_config,
            GeomCA_params=geomCA_config,
            loggers=exp_loggers,
        )

        # Evaluate DCA
        Delaunay_graph = DCA_algorithm.fit(data_config)

        # Plot results
        # DCA_visualization._plot_UMAP_components(
        #     Delaunay_graph,
        #     experiment_id,
        #     os.path.join(self.trainer.default_root_dir, data_config.input_array_filepath),
        #     os.path.join(self.trainer.default_root_dir, hdbscan_config.input_array_labels_filepath),
        #     DCA_algorithm.visualization_dir,
        # )

        # DCA_visualization._plot_RE_components_consistency(
        #     Delaunay_graph,
        #     DCA_algorithm.visualization_dir,
        #     min_comp_size=2,
        #     annotate_largest=True,
        #     display_smaller=False,
        # )

        # DCA_visualization._plot_RE_components_quality(
        #     Delaunay_graph,
        #     DCA_algorithm.visualization_dir,
        #     min_comp_size=2,
        #     annotate_largest=True,
        #     display_smaller=False,
        # )

        # Extract metrics
        for stat, stat_value in Delaunay_graph.network_stats.__dict__.items():
            print(" ====> " + f"{stat}: {stat_value}")

        return Delaunay_graph

    def log_Delaunay_graph_stats(self, Delaunay_graph, dca_experiment_id):
        self.logger.add_log(
            f"{dca_experiment_id}_P", Delaunay_graph.network_stats.precision
        )
        self.logger.add_log(
            f"{dca_experiment_id}_R", Delaunay_graph.network_stats.recall
        )
        self.logger.add_log(
            f"{dca_experiment_id}_q", Delaunay_graph.network_stats.network_quality
        )
        self.logger.add_log(
            f"{dca_experiment_id}_c", Delaunay_graph.network_stats.network_consistency
        )

        # Component stats
        for comp_idx in range(Delaunay_graph.first_trivial_component_idx):
            self.logger.add_log(
                f"{dca_experiment_id}_component{comp_idx}_consistency",
                Delaunay_graph.comp_stats[comp_idx].comp_consistency,
            )
            self.logger.add_log(
                f"{dca_experiment_id}_component{comp_idx}_quality",
                Delaunay_graph.comp_stats[comp_idx].comp_quality,
            )
            self.logger.add_log(
                f"{dca_experiment_id}_component{comp_idx}_num_edges",
                Delaunay_graph.comp_stats[comp_idx].num_total_comp_edges,
            )
            self.logger.add_log(
                f"{dca_experiment_id}_component{comp_idx}_num_RE_edges",
                Delaunay_graph.comp_stats[comp_idx].num_comp_RE_edges,
            )
            self.logger.add_log(
                f"{dca_experiment_id}_component{comp_idx}_num_R",
                len(Delaunay_graph.comp_stats[comp_idx].Ridx),
            )
            self.logger.add_log(
                f"{dca_experiment_id}_component{comp_idx}_num_E",
                len(Delaunay_graph.comp_stats[comp_idx].Eidx),
            )

    def test_epoch_end(self, outputs):
        n_mod = len(list(outputs[0].keys()))
        R = torch.concat([outputs[i][-1] for i in range(len(outputs))]).cpu().numpy()
        E_repr = []
        for mod in range(n_mod - 1):
            print(f"Evaluated joint_m{mod}")
            E = (
                torch.concat([outputs[i][mod] for i in range(len(outputs))])
                .cpu()
                .numpy()
            )
            if mod in self.unique_modality_idxs:
                E = np.unique(E.round(4), axis=0)
                assert (
                    E.shape[0]
                    == self.unique_modality_dims[self.unique_modality_idxs.index(mod)]
                )
            E_repr.append(E)
            Delaunay_graph = self.evaluate(R, E, f"joint_m{mod}")
            self.log_Delaunay_graph_stats(Delaunay_graph, f"joint_m{mod}")
            del Delaunay_graph, E

        # # Extra eval
        # for mod0, mod1 in self.partial_modalities_idxs:
        #     Delaunay_graph = self.evaluate(
        #         E_repr[mod0], E_repr[mod1], f"m{mod0}_m{mod1}"
        #     )
        #     self.log_Delaunay_graph_stats(Delaunay_graph, f"m{mod0}_m{mod1}")

        num_halv_R = int(R.shape[0] / 2)
        Delaunay_graph = self.evaluate(R[:num_halv_R], R[num_halv_R:], f"joint_joint")
        self.log_Delaunay_graph_stats(Delaunay_graph, f"joint_joint")

    def evaluate_dca(self):
        start_time = time.time()
        self.model.eval()
        test_dataloader = self.data_module.test_dataloader()
        outputs = []
        for idx, batch in enumerate(test_dataloader):
            output = self.test_step(batch, idx)
            outputs.append(output)
        # evaluator
        self.test_epoch_end(outputs)
        print(f"DCA eval time: {time.time() - start_time}s")