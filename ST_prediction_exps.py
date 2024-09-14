
import sys,os,shutil,json,h5py,glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import openslide
import pickle
from sklearn.metrics import r2_score
import idr_torch
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist


CLASSIFICATION_DICT = {}
REGRESSION_LIST = []
with open('/data/zhongz2/temp29/debug/ST_gene_list.pkl', 'rb') as fp:
    gene_map_dict = {v: str(i) for i, v in enumerate(pickle.load(fp)['gene_list'])}
    REGRESSION_LIST = list(gene_map_dict.values())
BACKBONE_DICT = {
    'resnet50': 2048
}
GLOBAL_MEAN = [0.75225115, 0.5662438 , 0.72874427]
GLOBAL_STD = [0.12278537, 0.14380322, 0.10359251]


def test_vst_vis(): # fine
    sc.logging.print_versions()
    sc.set_figure_params(facecolor="white", figsize=(8, 8))
    sc.settings.verbosity = 3

    adata = sc.datasets.visium_sge(sample_id="CytAssist_11mm_FFPE_Human_Lung_Cancer")
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pl.spatial(adata, color="log1p_n_genes_by_counts", cmap="hsv", save=True)


# version = 'v0' # just RGB patch, normalize with imagenet-mean/std
def create_data():

    version = sys.argv[1]

    version = 'v2'
    spot_scale = 1.3
    # data_root = os.path.join('/data/zhongz2/temp_ST_prediction', f'data_{version}')
    data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], f'data_{version}_{spot_scale}')
    os.makedirs(data_root, exist_ok=True)

    root = '/data/zhongz2/ST_20240903'
    df = pd.read_excel(f'{root}/ST_20240907.xlsx')
    svs_dir = os.path.join(root, 'svs')
    patches_dir = os.path.join(root, 'patches')
    gene_vst_dir = os.path.join(root, 'gene_vst')

    human_slide_ids = {
        # '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Glioblastoma_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1',
        # '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1',
        # '10x_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_2.0.0',
        # '10x_CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_2.0.0',
        # '10x_CytAssist_FFPE_Protein_Expression_Human_Tonsil_2.1.0',
        # '10x_CytAssist_Fresh_Frozen_Human_Breast_Cancer_2.0.1',
        # '10x_Targeted_Visium_Human_BreastCancer_Immunology_1.2.0',
        # '10x_V1_Breast_Cancer_Block_A_Section_1_1.1.0',
        # '10x_V1_Breast_Cancer_Block_A_Section_2_1.1.0',
        # '10x_Visium_FFPE_Human_Cervical_Cancer_1.3.0',
        # '10x_Visium_FFPE_Human_Intestinal_Cancer_1.3.0',
        # '10x_Visium_FFPE_Human_Ovarian_Cancer_1.3.0',
        # '10x_Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma_1.3.0',
        # '10x_Visium_Human_Breast_Cancer_1.3.0',
        # 'ST1K4M_Human_Breast_10X_06092021_Visium',
        # 'ST1K4M_Human_Colon_10X_10052023_Visium_control_rep1',
        # 'ST1K4M_Human_Colon_10X_10052023_Visium_control_rep2',
        # 'ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep1',
        # 'ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep2',
        # 'ST1K4M_Human_Prostate_10X_06092021_Visium_cancer',
        # 'ST1K4M_Human_Prostate_10X_06092021_Visium_normal',
        # 'ST1K4M_Human_Prostate_10X_07122022_Visium'
    }
    df = df[df['slide_id'].isin(human_slide_ids)].reset_index(drop=True)

    # filename = os.path.join(data_root, 'all_gene_names.pkl')
    # if os.path.exists(filename):
    #     with open(filename, 'rb') as fp:
    #         all_gene_names = pickle.load(fp)['all_gene_names']
    # else:
    #     if idr_torch.world_size > 1:
    #         raise ValueError("error")
    #         sys.exit(-1)

    #     # get common genes
    #     gene_names = {}
    #     for rowid, row in df.iterrows():
    #         svs_prefix = row['slide_id']
    #         vst_filename_db = os.path.join(root, 'vst_dir_db', svs_prefix+'_original_VST.db')
    #         parquet_file = pq.ParquetFile(vst_filename_db)
    #         existing_columns = parquet_file.schema.names
    #         gene_names[svs_prefix] = [v for v in existing_columns if '__' != v[:2]]
    #     gene_names1 = list(gene_names.values())
    #     all_gene_names = sorted(list(set(gene_names1[0]).union(*gene_names1[1:])))
    #     with open(filename, 'wb') as fp:
    #         pickle.dump({'all_gene_names': all_gene_names}, fp)

    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size) 
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)

    global_mean = np.zeros(3, dtype=np.float32)
    global_std = np.zeros(3, dtype=np.float32)

    column_names_dict = {}
    for rowid, row in sub_df.iterrows():
        svs_prefix = row['slide_id']
        save_filename = os.path.join(data_root, svs_prefix+'.pkl')
        svs_filename = os.path.join(root, 'svs', svs_prefix+'.svs')
        vst_filename_db = os.path.join(root, 'vst_dir_db', svs_prefix+'_original_VST.db')
        parquet_file = pq.ParquetFile(vst_filename_db)
        existing_columns = parquet_file.schema.names
        meta_columns = ['__barcode', '__spot_X', '__spot_Y', '__upperleft_X', '__upperleft_Y']

        query_columns = ['__spot_X', '__spot_Y']
        xy_df = pd.read_parquet(vst_filename_db, columns=query_columns)
        query_columns = [col for col in existing_columns if '__' != col[:2]]
        vst_df = pd.read_parquet(vst_filename_db, columns=query_columns)
        vst_df = vst_df.clip(lower=-8, upper=8, axis=1)
        vst_df = vst_df.rename(columns=gene_map_dict)
        column_names_dict[svs_prefix] = vst_df.columns.values.tolist()

        spot_size = row['spot_size']
        patch_size = int(np.ceil(spot_scale * spot_size)) # expand some area (10% here)
        st_patch_size = patch_size
        slide = openslide.open_slide(svs_filename)

        save_dir = os.path.join(data_root, svs_prefix)
        os.makedirs(save_dir, exist_ok=True)
        mean = np.zeros(3, dtype=np.float32)
        std = np.zeros(3, dtype=np.float32)
        for (_, row1), (_, row2) in zip(xy_df.iterrows(), vst_df.iterrows()):
            x, y = int(row1['__spot_X'])-st_patch_size//2, int(row1['__spot_Y'])-st_patch_size//2  # left, top
            patch = slide.read_region(location=(x,y), level=0, size=(st_patch_size, st_patch_size)).convert('RGB')
            save_filename = os.path.join(save_dir, f'x{x}_y{y}.jpg')
            patch.save(save_filename)
            with open(save_filename.replace('.jpg', '.txt'), 'w') as fp:
                fp.write(','.join(['{:.4f}'.format(v) for v in row2.values.tolist()]))

            patch = np.array(patch)/255
            mean += patch.mean((0, 1))
            std += patch.std((0, 1)) 
            
        mean /= len(vst_df)
        std /= len(vst_df)

        global_mean += mean
        global_std += std
        print(svs_prefix)

    global_mean /= len(sub_df)
    global_std /= len(sub_df)
    with open(os.path.join(data_root, 'mean_std.pkl'), 'wb') as fp:
        pickle.dump({'global_mean': global_mean, 'global_std': global_std}, fp)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def ddp_setup():
    dist.init_process_group(backend="nccl")


def collect_results_gpu(part_tensor, size, world_size):
    shape = part_tensor.shape
    shape_tensor = torch.tensor(shape[0], device=part_tensor.device)
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max()

    if len(shape) == 1:
        part_send = torch.zeros(shape_max, dtype=part_tensor.dtype, device=part_tensor.device)
        part_send[:shape_tensor] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        dist.all_gather(part_recv_list, part_send)
    if len(shape) == 2:
        part_send = torch.zeros((shape_max, shape[1]), dtype=part_tensor.dtype, device=part_tensor.device)
        part_send[:shape_tensor] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max, shape[1]) for _ in range(world_size)
        ]
        dist.all_gather(part_recv_list, part_send)
    return torch.cat(part_recv_list, axis=0)[:size]


class Trainer:
    def __init__(
            self,
            model,
            dataloaders,
            optimizer,
            class_weights_dict={},
            save_root='/data/zhongz2/temp_ST_prediction/outputs',
            save_every=1,
            accum_iter=8,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.model = model.to(self.gpu_id)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.epochs_run = 0
        self.snapshot_path = os.path.join(save_root, 'snapshot.pt') 
        self.accum_iter = accum_iter
        self.save_every = save_every
        self.save_root = save_root

        cls_loss_fn_dict, reg_loss_fn_dict = get_loss_fn_dict(class_weights_dict, device=self.gpu_id)
        self.cls_loss_fn_dict = cls_loss_fn_dict
        self.reg_loss_fn_dict = reg_loss_fn_dict

        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.loss_dicts = {subset: [] for subset in self.dataloaders.keys()}

        if self.gpu_id == 0:
            save_dirs = {}
            for subset in self.dataloaders.keys():
                save_dirs[subset] = os.path.join(save_root, subset)
                os.makedirs(save_dirs[subset], exist_ok=True)
            self.save_dirs = save_dirs

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_epoch(self, epoch, subset='train'):

        is_train = subset == 'train'
        if is_train:
            self.model.train()
            self.model.zero_grad()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        b_sz = len(next(iter(self.dataloaders[subset]))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.dataloaders[subset])}")
        self.dataloaders[subset].sampler.set_epoch(epoch)
        dataset = self.dataloaders[subset].dataset
        results = {}
        labels = {}
        all_svs_filenames = []
        for batch_idx, (patches, labeled_batch) in enumerate(self.dataloaders[subset]):
            patches = patches.to(self.gpu_id)
            for k in labeled_batch.keys():
                if k != 'svs_filename':
                    labeled_batch[k] = labeled_batch[k].to(self.gpu_id)
                    if k in labels:
                        labels[k].append(labeled_batch[k].detach().clone())
                    else:
                        labels[k] = [labeled_batch[k].detach().clone()]
                
            if is_train:
                results_dicts = self.model(patches)
            else:
                with torch.no_grad():
                    results_dicts = self.model(patches)

                for k, v in results_dicts.items():
                    if k in results:
                        results[k].append(v.detach().clone())
                    else:
                        results[k] = [v.detach().clone()]

            classification_losses = {}
            for k in CLASSIFICATION_DICT.keys():
                if k in labeled_batch:
                    classification_losses[k] = self.cls_loss_fn_dict[k](results_dicts[k + '_logits'], labeled_batch[k])

            regression_losses = {}
            for k in REGRESSION_LIST:
                if k in labeled_batch:
                    regression_losses[k] = self.reg_loss_fn_dict[k](results_dicts[k + '_logits'], labeled_batch[k])

            total_loss = sum([v for vi, v in enumerate(classification_losses.values())]) + \
                         sum([v for vi, v in enumerate(regression_losses.values())])

            if is_train:

                total_loss = total_loss / self.accum_iter
                total_loss.backward()

                if (batch_idx + 1) % self.accum_iter == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()

        if not is_train:
            for k in results.keys():
                results[k] = torch.cat(results[k])
            for k in labels.keys():
                labels[k] = torch.cat(labels[k])

            all_results = {}
            for k in results.keys():
                all_results[k] = collect_results_gpu(results[k], len(dataset), world_size=self.world_size)
            all_labels = {}
            for k in labels.keys():
                all_labels[k] = collect_results_gpu(labels[k], len(dataset), world_size=self.world_size)

            if self.gpu_id == 0:
                loss_dict = self._save_results(epoch, all_labels, all_results, dataset, save_dir=self.save_dirs[subset],
                                            writer=None, subset=subset)
                self.loss_dicts[subset].append(loss_dict)

                for subset, v in self.loss_dicts.items():
                    if len(v) > 0:
                        val_log_df = pd.DataFrame(v)
                        val_log_df.to_csv(os.path.join(self.save_root, '{}_e{}_log.csv'.format(subset, epoch)))

        dist.barrier()

    def _save_results(self, epoch, all_labels, all_results, dataset, save_dir, writer=None, subset='val'):
        losses_dict = {
            'bce': 0., 'kld': 0.,
            'surv': 0., 'regu': 0.
        }
        for k in CLASSIFICATION_DICT.keys():
            losses_dict[k] = 0.
        for k in REGRESSION_LIST:
            losses_dict[k] = 0.

        loggers_dict = {}
        cls_invalids = {k: False for k in CLASSIFICATION_DICT.keys()}
        for k, v in CLASSIFICATION_DICT.items():
            if k not in dataset.classification_dict:
                cls_invalids[k] = True
                continue
            Y = all_labels[k]
            if len(Y[torch.where(Y != IGNORE_INDEX_DICT[k])[0]].unique()) < 2:
                cls_invalids[k] = True
                continue
            loggers_dict[k] = Accuracy_Logger(n_classes=len(v), task_name=k, label_names=v,
                                              ignore_label_ind=IGNORE_INDEX_DICT[k])
            Y_hat_k = torch.topk(all_results[k + '_logits'], 1, dim=1)[1].squeeze(1)
            Y_prob_k = F.softmax(all_results[k + '_logits'], dim=1)
            loggers_dict[k].log(Y_hat_k, Y, Y_prob_k)
            losses_dict[k] += self.cls_loss_fn_dict[k](all_results[k + '_logits'], Y).item()

        reg_loggers_dict = {}
        reg_invalids = {k: False for k in REGRESSION_LIST}
        for k in REGRESSION_LIST:
            if k not in dataset.regression_list:
                reg_invalids[k] = True
                continue
            Y = all_labels[k]
            if len(Y) < 2:
                reg_invalids[k] = True
                continue
            reg_loggers_dict[k] = Regression_Logger()
            reg_loggers_dict[k].log(all_results[k + '_logits'], Y)
            losses_dict[k] += self.reg_loss_fn_dict[k](all_results[k + '_logits'], Y).item()

        for name, labels in CLASSIFICATION_DICT.items():
            if name not in dataset.classification_dict:
                continue
            if cls_invalids[name]:
                continue
            loggers_dict[name].set_save_filename(
                os.path.join(save_dir, 'epoch_{:03}_{}_{}_data.txt'.format(epoch, name, subset)))

            for average in ['micro', 'macro', 'weighted']:
                score = loggers_dict[name].get_f1_score(average=average)
                losses_dict['{}_f1_{}'.format(name, average)] = score
                if writer:
                    writer.add_scalar('{}/{}_f1_{}'.format(subset, name, average), score, epoch)

            for average in ['macro', 'weighted']:
                auc = loggers_dict[name].get_auc_score(average=average)
                losses_dict['{}_auc_{}'.format(name, average)] = auc
                if writer:
                    writer.add_scalar('{}/{}_auc_{}'.format(subset, name, average), auc, epoch)

            loggers_dict[name].get_roc_curve(
                os.path.join(save_dir, 'epoch_{:03}_{}_{}_ROC.jpg'.format(epoch, name, subset)))

            loggers_dict[name].save_data(
                os.path.join(save_dir, 'epoch_{:03}_{}_{}_data.txt'.format(epoch, name, subset)))

            loggers_dict[name].get_classification_report(
                os.path.join(save_dir, 'epoch_{:03}_{}_{}_report.txt'.format(epoch, name, subset)))

            for j in range(len(labels)):
                acc, correct, count = loggers_dict[name].get_summary(j)
                losses_dict['{}_{}_acc'.format(name, CLASSIFICATION_DICT[name][j])] = acc
                losses_dict['{}_{}_correct'.format(name, CLASSIFICATION_DICT[name][j])] = correct
                losses_dict['{}_{}_count'.format(name, CLASSIFICATION_DICT[name][j])] = count
                if writer:
                    writer.add_scalar('{}/{}_{}_{}_acc'.format(subset, name, j, CLASSIFICATION_DICT[name][j]), acc,
                                      epoch)

        for name in REGRESSION_LIST:
            if name not in dataset.regression_list:
                continue
            if reg_invalids[name]:
                continue
            mse = reg_loggers_dict[name].mean_squared_error()
            losses_dict['{}_mse'.format(name)] = mse
            if writer:
                writer.add_scalar('{}/{}_mse'.format(subset, name), mse, epoch)

            reg_metrics_dict = reg_loggers_dict[name].compute_metrics()
            for metric_name, metric_val in reg_metrics_dict.items():
                losses_dict['{}_{}'.format(name, metric_name)] = metric_val
                if writer:
                    writer.add_scalar('{}/{}_{}'.format(subset, name, metric_name), metric_val, epoch)

        losses_dict['epoch'] = epoch 

        return losses_dict

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path.replace('.pt', '_{}.pt'.format(epoch)))
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):

            for subset in self.dataloaders.keys():
                if self.dataloaders[subset] is not None:
                    self._run_epoch(epoch, subset)

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

        if self.gpu_id == 0:
            for subset, v in self.loss_dicts.items():
                if len(v) > 0:
                    val_log_df = pd.DataFrame(v)
                    val_log_df.to_csv(os.path.join(self.save_root, '{}_log.csv'.format(subset)))


def get_loss_fn_dict(class_weights_dict, device):
    cls_loss_fn_dict = {}
    for k in CLASSIFICATION_DICT.keys():
        cls_loss_fn_dict[k] = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_DICT[k],
                                                  weight=class_weights_dict[k].to(device))
    reg_loss_fn_dict = {k: nn.MSELoss() for k in REGRESSION_LIST}

    return cls_loss_fn_dict, reg_loss_fn_dict


def get_class_weights(train_datasets, num_classes, label_col):
    Ns = 0
    slide_cls_ids = [0 for _ in range(num_classes)]
    for train_dataset in train_datasets:
        Ns += float(len(train_dataset))
        for i in range(num_classes):
            slide_cls_ids[i] += len(np.where(train_dataset.slide_data[label_col] == i)[0])
    weight_per_class = [Ns / slide_cls_ids[c] if slide_cls_ids[c] > 0 else 0 for c in range(num_classes)]
    class_weights = torch.FloatTensor(weight_per_class)
    return class_weights




class STModel(nn.Module):
    def __init__(self, backbone='resnet50', dropout=0.25):
        super().__init__()

        if backbone == 'resnet50':
            self.backbone_model = torchvision.models.resnet50(pretrained=True)
            self.backbone_model.fc = nn.Identity()
        else:
            raise ValueError('error')

        self.rho = nn.Sequential(*[
            nn.Linear(BACKBONE_DICT[backbone], 512), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout)
        ])

        classifiers = {}
        for k, labels in CLASSIFICATION_DICT.items():
            classifiers[k] = nn.Linear(256, len(labels))
        self.classifiers = nn.ModuleDict(classifiers)
        regressors = {}
        for k in REGRESSION_LIST:
            regressors[k] = nn.Linear(256, 1)
        self.regressors = nn.ModuleDict(regressors)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        h = self.backbone_model(x)

        h = self.rho(h)

        results_dict = {}
        for k, classifier in self.classifiers.items():
            logits_k = classifier(h)
            results_dict[k + '_logits'] = logits_k

        for k, regressor in self.regressors.items():
            values_k = regressor(h).squeeze(1)
            results_dict[k + '_logits'] = values_k

        return results_dict



class PatchDataset1(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform
        self.classification_dict = {}
        self.regression_list = list(self.gene_map_dict.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        patch = Image.open(self.data[idx])
        with open(self.data[idx].replace('.jpg', '.txt'), 'r') as fp:
            labels = [float(v) for v in fp.readline().split(',')]
        return self.transform(patch), labels


def load_train_objs(): 
    # data
    version = 'v2'
    spot_scale = 1.3
    # data_root = os.path.join('/data/zhongz2/temp_ST_prediction', f'data_{version}')
    data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], f'data_{version}_{spot_scale}')

    invalid_prefixes = [
        'selected_gene_names', 'mean_std'
    ]
    val_prefixes = [
        # '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1'
        '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1'
    ]
    train_data = []
    val_data = []
    # files = sorted(glob.glob(os.path.join(data_root, '*.pkl')))
    files = sorted(os.listdir(data_root))
    for d in files:
        if not os.path.isdir(os.path.join(data_root, d)):
            print(d, 'not dir')
            continue
        svs_prefix = d
        if svs_prefix in invalid_prefixes:
            continue
        samples = glob.glob(os.path.join(data_root, d, '*.jpg'))
        if svs_prefix in val_prefixes:
            val_data.extend(samples) 
        else:
            train_data.extend(samples) 

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = GLOBAL_MEAN
    std = GLOBAL_STD
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = PatchDataset1(train_data, transform=train_transform)
    val_dataset = PatchDataset1(val_data, transform=val_transform)

    model = STModel()

    # Freeze the parameters of the pre-trained layers
    for param in model.backbone_model.parameters():
        param.requires_grad = False
        
    lr = 1e-4
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return train_dataset, val_dataset, model, optimizer


def train_main():
    max_epochs = 20
    ddp_setup()
    class_weights_dict = {}
    train_dataset, val_dataset, model, optimizer = load_train_objs()

    dataloaders = {
        'train':
            DataLoader(train_dataset, num_workers=4, batch_size=1, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False)),
        'val':
            DataLoader(val_dataset, num_workers=4, batch_size=1, pin_memory=True, shuffle=False, sampler=DistributedSampler(val_dataset, shuffle=False, drop_last=False)),
    }
    trainer = Trainer(model, dataloaders, optimizer, class_weights_dict, save_root='/lscratch/'+os.environ['SLURM_JOB_ID']+'/outputs', save_every=2, accum_iter=2)
    trainer.train(max_epochs=max_epochs)
    dist.destroy_process_group()


if __name__ == '__main__':
    train_main()


"""

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29898 \
    ST_prediction_exps.py

"""

def main():

    version = 'v1'
    max_epochs = 100
    device = torch.device('cuda:0')
    # data
    version = 'v2'
    spot_scale = 1.3
    # data_root = os.path.join('/data/zhongz2/temp_ST_prediction', f'data_{version}')
    data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], f'data_{version}_{spot_scale}')

    invalid_prefixes = [
        'selected_gene_names', 'mean_std'
    ]
    val_prefixes = [
        '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1'
    ]
    train_data = []
    val_data = []
    # files = sorted(glob.glob(os.path.join(data_root, '*.pkl')))
    files = sorted(os.listdir(data_root))
    for d in files:
        if not os.path.isdir(os.path.join(data_root, d)):
            print(d, 'not dir')
            continue
        svs_prefix = d
        if svs_prefix in invalid_prefixes:
            continue
        samples = glob.glob(os.path.join(data_root, d, '*.pkl'))
        if svs_prefix in val_prefixes:
            val_data.extend(samples) 
        else:
            train_data.extend(samples) 

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = GLOBAL_MEAN
    std = GLOBAL_STD
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = PatchDataset1(train_data, transform=train_transform)
    val_dataset = PatchDataset1(val_data, transform=val_transform)

    model = STModel()

    # Freeze the parameters of the pre-trained layers
    for param in model.backbone_model.parameters():
        param.requires_grad = False
        
    lr = 1e-4
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    train_loader = DataLoader(train_dataset, num_workers=2, batch_size=1, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, num_workers=2, batch_size=1, shuffle=False, drop_last=False)

    cls_loss_fn_dict = {}
    reg_loss_fn_dict = {k: nn.MSELoss() for k in REGRESSION_LIST}

    log_strs = []
    labels = {}
    results = {}
    is_train = True
    accum_iter = 4
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    for batch_idx, (patches, labeled_batch) in enumerate(train_loader):
        patches = patches.to(device)
        for k in labeled_batch.keys():
            if k != 'svs_filename':
                labeled_batch[k] = labeled_batch[k].to(device)
                if k in labels:
                    labels[k].append(labeled_batch[k].detach().clone())
                else:
                    labels[k] = [labeled_batch[k].detach().clone()]
            
        if is_train:
            results_dicts = self.model(patches)
        else:
            with torch.no_grad():
                results_dicts = self.model(patches)

        for k, v in results_dicts.items():
            if k in results:
                results[k].append(v.detach().clone())
            else:
                results[k] = [v.detach().clone()]

        regression_losses = {}
        for k in REGRESSION_LIST:
            if k in labeled_batch:
                regression_losses[k] = reg_loss_fn_dict[k](results_dicts[k + '_logits'], labeled_batch[k])

        total_loss = sum([v for vi, v in enumerate(classification_losses.values())]) + \
                        sum([v for vi, v in enumerate(regression_losses.values())])

        if is_train:

            total_loss = total_loss / accum_iter
            total_loss.backward()

            if (batch_idx + 1) % accum_iter == 0:
                self.optimizer.step()
                self.model.zero_grad()
                self.optimizer.zero_grad()
    
