from src.data.openbhb.dataset import OpenBHBDataset

# DataModule requires lightning — import directly when needed to avoid
# triggering the torchmetrics→transformers→huggingface_hub version chain.
#   from src.data.openbhb.datamodule import OpenBHBDataModule
