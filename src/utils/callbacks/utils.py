from pytorch_lightning.callbacks import Callback

class RunTestEveryNEpochs(Callback):
    def __init__(self, every_n_epoch: int):
        self.n = every_n_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        # Kiểm tra xem epoch hiện tại có chia hết cho n không
        if (trainer.current_epoch + 1) % self.n == 0:
            print(f"\n--- Running Test at epoch Epoch {trainer.current_epoch} ---")
            # Gọi trực tiếp tập test
            # (Bạn cần truyền đúng test_dataloader của mình vào đây)
            trainer.test(model=pl_module, dataloaders=trainer.datamodule.test_dataloader())