import torch
import time
import pickle
from tqdm import tqdm
from util_scripts.wandb_logger import WandbLogger
from util_scripts.train_callbacks import ModelSaverLoaderCallback

class ModelTrainer():
    def __init__(self, model, dataset, data_module, opt):
        self.opt = opt
        self.model = model
        self.dataset = dataset
        self.data_module = data_module
        self.device = opt.device
        # optimizer and scheduler
        self.optimizer, self.scheduler = self.configure_optimizers()
        # logger
        self.logger = WandbLogger(opt)
        # callback
        self.callback = ModelSaverLoaderCallback(opt.result_path, 'model', opt=opt)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opt.learning_rate)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return optimizer, scheduler
    
    def training_step(self, batch, epoch, out_uncertainty=False, beta=1.):
        # Forward pass through the encoders
        if self.dataset in ['mosi', 'mosei']:
            batch_X, batch_Y, _ = batch[0], batch[1], batch[2]

            _, text, audio, vision = batch_X
            target_data, mask_data = batch_Y
            target_data = target_data.squeeze(-1).to(self.device)
            mask_data = mask_data.squeeze(-1).to(self.device)
            # data = [text.to(self.device), audio.to(self.device), vision.to(self.device)]
            data = [text.to(self.device), audio.to(self.device)]
        elif self.dataset in ['mmimdb', 'food101', 'hatememes']:
            batch_X, batch_Y = batch
            image, text = batch_X
            target_data, mask_data = batch_Y

            target_data = target_data.float().squeeze(-1).to(self.device)
            mask_data = mask_data.squeeze(-1).to(self.device)
            data = [text.to(self.device), image.to(self.device)]
        elif self.dataset in ['book']:
            if self.opt.use_fast_loading:
                txt, img, att, target, mask = batch
                target_data = target.to(self.device)
                mask_data = mask.unsqueeze(-1).unsqueeze(-1).to(self.device)
                data = [txt.to(self.device), img.to(self.device), att.to(self.device), mask_data]
            else:
                text, segment, mask, image, target_data, mask_data = batch
                target_data = target_data.to(self.device)
                mask_data = mask_data.unsqueeze(-1).unsqueeze(-1).to(self.device)
                data = [text.to(self.device), mask.to(self.device), segment.to(self.device), image.to(self.device), mask_data]
        elif self.dataset in ['mmml_mosi']:
            text_inputs = batch["reps_T"].to(self.device)
            audio_inputs = batch["reps_A"].to(self.device)
            idx_data = batch["index"].to(self.device)

            target_data = batch["label"].to(self.device)
            mask_data = batch["sample_mask"].unsqueeze(-1).to(self.device)
            data = [text_inputs, audio_inputs, mask_data, idx_data]
        elif self.dataset in ['mmact', 'utd_mhad']:
            vid, accel, gyro, target, mask = batch
            target_data = target.to(self.device)
            mask_data = mask.to(self.device)
            data = [vid.to(self.device), accel.to(self.device), gyro.to(self.device), mask_data]

        # Forward
        loss, tqdm_dict = self.model.training_step(data, target_data, mask_data, self.opt, epoch, out_uncertainty, beta)

        output = {
            "loss": loss,
            "log": tqdm_dict
        }
        return output
    
    def training_epoch_end(self, epoch, outputs):
        log_keys = list(outputs[0]["log"].keys())
        for log_key in log_keys:
            avg_batch_log = (
                torch.stack(
                    [
                        outputs[batch_output_idx]["log"][log_key]
                        for batch_output_idx in range(len(outputs))
                    ]
                )
                .mean()
            )
            self.logger.add_log(f"train/{log_key}", avg_batch_log)

        self.logger.write_log(epoch)

    def validation_step(self, batch, epoch, out_uncertainty=False, beta=1.):
        # Forward pass through the encoders
        if self.dataset in ['mosi', 'mosei']:
            batch_X, batch_Y, _ = batch[0], batch[1], batch[2]

            _, text, audio, vision = batch_X
            target_data, mask_data = batch_Y
            target_data = target_data.squeeze(-1).to(self.device)
            mask_data = mask_data.squeeze(-1).to(self.device)
            # data = [text.to(self.device), audio.to(self.device), vision.to(self.device)]
            data = [text.to(self.device), audio.to(self.device)]
        elif self.dataset in ['mmimdb', 'food101', 'hatememes']:
            batch_X, batch_Y = batch
            image, text = batch_X
            target_data, mask_data = batch_Y

            target_data = target_data.float().squeeze(-1).to(self.device)
            mask_data = mask_data.squeeze(-1).to(self.device)
            data = [text.to(self.device), image.to(self.device)]
        elif self.dataset in ['book']:
            if self.opt.use_fast_loading:
                txt, img, att, target, mask = batch
                target_data = target.to(self.device)
                mask_data = mask.unsqueeze(-1).unsqueeze(-1).to(self.device)
                data = [txt.to(self.device), img.to(self.device), att.to(self.device), mask_data]
            else:
                text, segment, mask, image, target_data, mask_data = batch
                target_data = target_data.to(self.device)
                mask_data = mask_data.unsqueeze(-1).unsqueeze(-1).to(self.device)
                data = [text.to(self.device), mask.to(self.device), segment.to(self.device), image.to(self.device), mask_data]
        elif self.dataset in ['mmml_mosi']:
            text_inputs = batch["reps_T"].to(self.device)
            audio_inputs = batch["reps_A"].to(self.device)
            idx_data = batch["index"].to(self.device)

            target_data = batch["label"].to(self.device)
            mask_data = batch["sample_mask"].unsqueeze(-1).to(self.device)
            data = [text_inputs, audio_inputs, mask_data, idx_data]
        elif self.dataset in ['mmact', 'utd_mhad']:
            vid, accel, gyro, target, mask = batch
            target_data = target.to(self.device)
            mask_data = mask.to(self.device)
            data = [vid.to(self.device), accel.to(self.device), gyro.to(self.device), mask_data]
        
        output_dict = self.model.validation_step(data, target_data, mask_data, self.opt, epoch, out_uncertainty, beta)
        return output_dict
    
    def validation_epoch_end(self, epoch, outputs):
        log_keys = list(outputs[0].keys())
        for log_key in log_keys:
            avg_batch_log = (
                torch.stack(
                    [
                        outputs[batch_output_idx][log_key]
                        for batch_output_idx in range(len(outputs))
                    ]
                )
                .mean()
            )
            self.logger.add_log(f"val/{log_key}", avg_batch_log)
            if log_key == self.opt.ckpt_metric:
                self.callback.save_cpkt(self.model, avg_batch_log)

        self.logger.write_log(epoch)
    
    def reset(self):
        self.optimizer, self.scheduler = self.configure_optimizers()
        self.callback.reset('loss')
        self.opt.ckpt_metric = 'loss'

    def fit(self):
        if self.opt.save_embedding:
            self.save_embedding()
            return
        if self.opt.dataset in ['mosi', 'mosei', 'mmml_mosi']:
            self.fit_mosi()
        elif self.opt.dataset in ['book']:
            self.fit_book()
        elif self.opt.dataset in ['mmact', 'utd_mhad']:
            self.fit_mmact()

    def fit_book(self):
        start_time = time.time()
        
        # # major training
        # for epoch in range(self.opt.n_epochs):
        #     # Train
        #     self.model.train()
        #     outputs, val_outputs = [], []
        #     for batch in tqdm(self.data_module.train_dataloader()):
        #         self.optimizer.zero_grad()
        #         output = self.training_step(batch, epoch)
        #         outputs.append(output)

        #         if output["loss"] != 0:
        #             output["loss"].backward()
        #             self.optimizer.step()

        #     self.training_epoch_end(epoch, outputs)
            
        #     # Validation
        #     self.model.eval()
        #     with torch.no_grad():
        #         for batch in self.data_module.val_dataloader():
        #             output = self.validation_step(batch, epoch)
        #             val_outputs.append(output)

        #     self.validation_epoch_end(epoch, val_outputs)
        #     self.scheduler.step(output["loss"])

        # train output uncertainty
        for param in self.model.reconstructor_shared.parameters():
            param.requires_grad = False
        for param in self.model.reconstructor_muys.parameters():
            param.requires_grad = False
        for param in self.model.reconstructor_sigma2s.parameters():
            param.requires_grad = False
        # for param in self.model.model.preoutput_fused_layers.parameters():
        #     param.requires_grad = False
        # for param in self.model.model.parameters():
        #     param.requires_grad = False
        # for param in self.model.model.fused_uncertainty.parameters():
        #     param.requires_grad = True
        self.reset()
        print('Train output uncertainty' + '-'*50)

        for epoch in range(self.opt.n_epochs):
            # Train
            self.model.train()
            outputs, val_outputs = [], []
            for batch in tqdm(self.data_module.train_dataloader()):
            # for batch in tqdm(self.data_module.test_dataloader()):
                output = self.training_step(batch, epoch, out_uncertainty=True, beta=self.opt.beta)
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                outputs.append(output)

                output["loss"].backward()
                self.optimizer.step()

            self.training_epoch_end(epoch, outputs)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                # for batch in self.data_module.train_dataloader():
                for batch in self.data_module.val_dataloader():
                # for batch in self.data_module.test_dataloader():
                    output = self.validation_step(batch, epoch, out_uncertainty=True, beta=self.opt.beta)
                    val_outputs.append(output)
                
            self.validation_epoch_end(epoch, val_outputs)

        print(f"Training time: {time.time() - start_time}s")

    def fit_mmact(self):
        start_time = time.time()
        # # major training
        # for epoch in range(self.opt.n_epochs):
        #     # Train
        #     self.model.train()
        #     outputs, val_outputs = [], []
        #     for batch in tqdm(self.data_module.train_dataloader()):
        #         self.optimizer.zero_grad()
        #         output = self.training_step(batch, epoch)
        #         outputs.append(output)

        #         output["loss"].backward()
        #         self.optimizer.step()

        #     self.training_epoch_end(epoch, outputs)
            
        #     # Validation
        #     self.model.eval()
        #     with torch.no_grad():
        #         for batch in self.data_module.val_dataloader():
        #             output = self.validation_step(batch, epoch)
        #             val_outputs.append(output)
                
        #     self.validation_epoch_end(epoch, val_outputs)
        #     self.scheduler.step(output["loss"])
        
        # train output uncertainty
        for param in self.model.model.reconstructor_muys.parameters():
            param.requires_grad = False
        for param in self.model.model.reconstructor_sigma2s.parameters():
            param.requires_grad = False
        # for param in self.model.model.parameters():
        #     param.requires_grad = False
        # for param in self.model.model.joint_uncertainty.parameters():
        #     param.requires_grad = True
        self.reset()
        print('Train output uncertainty' + '-'*50)

        for epoch in range(self.opt.n_epochs):
            # Train
            self.model.train()
            outputs, val_outputs = [], []
            for batch in tqdm(self.data_module.train_dataloader()):
            # for batch in tqdm(self.data_module.test_dataloader()):
                output = self.training_step(batch, epoch, out_uncertainty=True, beta=self.opt.beta)
                self.optimizer.zero_grad()
                outputs.append(output)

                output["loss"].backward()
                self.optimizer.step()

            self.training_epoch_end(epoch, outputs)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                # for batch in self.data_module.train_dataloader():
                for batch in self.data_module.val_dataloader():
                # for batch in self.data_module.test_dataloader():
                    output = self.validation_step(batch, epoch, out_uncertainty=True, beta=self.opt.beta)
                    val_outputs.append(output)
                
            self.validation_epoch_end(epoch, val_outputs)
        
        print(f"Training time: {time.time() - start_time}s")

    def fit_mosi(self):
        start_time = time.time()
        
        # # major training
        # for epoch in range(self.opt.n_epochs):
        #     # Train
        #     self.model.train()
        #     outputs, val_outputs = [], []
        #     for batch in tqdm(self.data_module.train_dataloader()):
        #         self.optimizer.zero_grad()
        #         output = self.training_step(batch, epoch)
        #         outputs.append(output)

        #         output["loss"].backward()
        #         self.optimizer.step()

        #     self.training_epoch_end(epoch, outputs)
            
        #     # Validation
        #     self.model.eval()
        #     with torch.no_grad():
        #         for batch in self.data_module.val_dataloader():
        #             output = self.validation_step(batch, epoch)
        #             val_outputs.append(output)
                
        #     self.validation_epoch_end(epoch, val_outputs)
        #     self.scheduler.step(output["loss"])

        # train output uncertainty
        for param in self.model.model.reconstruct_muy_T.parameters():
            param.requires_grad = False
        for param in self.model.model.reconstruct_muy_A.parameters():
            param.requires_grad = False
        for param in self.model.model.reconstruct_sigma2_T.parameters():
            param.requires_grad = False
        for param in self.model.model.reconstruct_sigma2_A.parameters():
            param.requires_grad = False
        # for param in self.model.model.parameters():
        #     param.requires_grad = False
        # for param in self.model.model.fused_uncertainty.parameters():
        #     param.requires_grad = True
        self.reset()
        print('Train output uncertainty' + '-'*50)

        for epoch in range(self.opt.n_epochs):
            # Train
            self.model.train()
            outputs, val_outputs = [], []
            for batch in tqdm(self.data_module.train_dataloader()):
            # for batch in tqdm(self.data_module.test_dataloader()):
                output = self.training_step(batch, epoch, out_uncertainty=True, beta=self.opt.beta)
                self.optimizer.zero_grad()
                outputs.append(output)

                output["loss"].backward()
                self.optimizer.step()

            self.training_epoch_end(epoch, outputs)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                # for batch in self.data_module.train_dataloader():
                for batch in self.data_module.val_dataloader():
                # for batch in self.data_module.test_dataloader():
                    output = self.validation_step(batch, epoch, out_uncertainty=True, beta=self.opt.beta)
                    val_outputs.append(output)
                
            self.validation_epoch_end(epoch, val_outputs)
        
        print(f"Training time: {time.time() - start_time}s")

    def save_embedding(self):
        if self.opt.dataset in ['book']:
            print('Save embedding' + '-'*50)
            self.model.eval()
            with torch.no_grad():
                # train
                img_ebds, txt_ebds, att_datas, target_datas = [], [], [], []
                for batch in tqdm(self.data_module.train_dataloader()):
                    text, segment, mask, image, target_data, mask_data = batch
                    target_datas.append(target_data)
                    mask_data = mask_data.unsqueeze(-1).unsqueeze(-1).to(self.device)
                    data = [text.to(self.device), mask.to(self.device), segment.to(self.device), image.to(self.device), mask_data]
                    
                    img_representation, txt_reprensentation, attn = self.model.save_embedding(data)
                    img_ebds.append(img_representation)
                    txt_ebds.append(txt_reprensentation)
                    att_datas.append(attn)

                pickle.dump(
                    {'img': torch.cat(img_ebds, dim=0), 
                    'txt': torch.cat(txt_ebds, dim=0), 
                    'att': torch.cat(att_datas, dim=0), 
                    'target': torch.cat(target_datas, dim=0)}, open(f'{self.opt.root_path}/embedding_train.pkl', 'wb'))
                # val
                img_ebds, txt_ebds, att_datas, target_datas = [], [], [], []
                for batch in tqdm(self.data_module.val_dataloader()):
                    text, segment, mask, image, target_data, mask_data = batch
                    target_datas.append(target_data)
                    mask_data = mask_data.unsqueeze(-1).unsqueeze(-1).to(self.device)
                    data = [text.to(self.device), mask.to(self.device), segment.to(self.device), image.to(self.device), mask_data]
                    
                    img_representation, txt_reprensentation, attn = self.model.save_embedding(data)
                    img_ebds.append(img_representation)
                    txt_ebds.append(txt_reprensentation)
                    att_datas.append(attn)
                
                pickle.dump(
                    {'img': torch.cat(img_ebds, dim=0), 
                    'txt': torch.cat(txt_ebds, dim=0), 
                    'att': torch.cat(att_datas, dim=0), 
                    'target': torch.cat(target_datas, dim=0)}, open(f'{self.opt.root_path}/embedding_val.pkl', 'wb'))
                
                # test
                img_ebds, txt_ebds, att_datas, target_datas = [], [], [], []
                for batch in tqdm(self.data_module.test_dataloader()):
                    text, segment, mask, image, target_data, mask_data = batch
                    target_datas.append(target_data)
                    mask_data = mask_data.unsqueeze(-1).unsqueeze(-1).to(self.device)
                    data = [text.to(self.device), mask.to(self.device), segment.to(self.device), image.to(self.device), mask_data]
                    
                    img_representation, txt_reprensentation, attn = self.model.save_embedding(data)
                    img_ebds.append(img_representation)
                    txt_ebds.append(txt_reprensentation)
                    att_datas.append(attn)
                
                pickle.dump(
                    {'img': torch.cat(img_ebds, dim=0), 
                    'txt': torch.cat(txt_ebds, dim=0), 
                    'att': torch.cat(att_datas, dim=0), 
                    'target': torch.cat(target_datas, dim=0)}, open(f'{self.opt.root_path}/embedding_test.pkl', 'wb'))