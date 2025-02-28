# -*- coding: utf-8 -*-
import datetime

import torch
from tpu_star.experiment import TorchGPUExperiment

from .metrics import cer, wer, string_accuracy


class OCRExperiment(TorchGPUExperiment):

    def calculate_metrics(self, gt_texts, outputs):
        pred_texts = []
        for encoded in outputs.argmax(2).data.cpu().numpy():
            pred_texts.append(self.ctc_labeling.decode(encoded))
        texts = [self.ctc_labeling.preprocess(text) for text in gt_texts]
        return {
            'cer': cer(pred_texts, texts),
            'wer': wer(pred_texts, texts),
            'acc': string_accuracy(pred_texts, texts),
        }

    def handle_one_batch(self, batch):
        lengths = batch['encoded_length'].to(self.device, dtype=torch.int32)
        encoded = batch['encoded'].to(self.device, dtype=torch.int32)
        outputs = self.model(batch['image'].to(
            self.device, dtype=torch.float32))

        preds_size = torch.IntTensor(
            [outputs.size(1)] * batch['encoded'].shape[0])
        preds = outputs.log_softmax(2).permute(1, 0, 2)

        loss = self.criterion(preds, encoded, preds_size, lengths)

        batch_metrics = self.calculate_metrics(batch['gt_text'], outputs)
        self.metrics.update(loss=loss.detach().cpu().item(), **batch_metrics)

        if self.is_train:
            loss.backward()
            self.optimizer_step()
            self.optimizer.zero_grad()
            self.scheduler.step()
    
    # Janky hack attempt
    # https://discuss.pytorch.org/t/lr-scheduler-onecyclelr-causing-tried-to-step-57082-times-the-specified-number-of-total-steps-is-57080/90083/7
    @classmethod
    def resume(
        cls,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        checkpoint_path,
        train_loader,
        valid_loader,
        n_epochs,
        neptune=None,
        seed=None,
        **kwargs,
    ):
        checkpoint = torch.load(checkpoint_path)
        experiment_state_dict = checkpoint['experiment_state_dict']
        neptune_state_dict = checkpoint['neptune_state_dict']

        experiment_name = 'R+' + experiment_state_dict['experiment_name']

        experiment = cls(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            rank=experiment_state_dict['rank'],
            seed=experiment_state_dict['seed'] if seed is None else seed,
            verbose=experiment_state_dict['verbose'],
            verbose_end=experiment_state_dict['verbose_end'],
            verbose_ndigits=experiment_state_dict['verbose_ndigits'],
            verbose_step=experiment_state_dict['verbose_step'],
            use_progress_bar=experiment_state_dict['use_progress_bar'],
            base_dir=experiment_state_dict['base_dir'],
            jupyters_path=experiment_state_dict['jupyters_path'],
            notebook_name=experiment_state_dict['notebook_name'],
            experiment_name=experiment_name,
            neptune=None,
            neptune_params=neptune_state_dict['params'],
            best_saving=experiment_state_dict['best_saving'],
            last_saving=experiment_state_dict['last_saving'],
            low_memory=experiment_state_dict.get('low_memory', True),
            **kwargs
        )

        experiment.epoch = experiment_state_dict['epoch']
        experiment.model.load_state_dict(checkpoint['model_state_dict'])
        experiment.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        sheduler_dict = checkpoint['scheduler_state_dict']
        sheduler_dict['total_steps'] = sheduler_dict['total_steps'] + n_epochs * int(len(train_loader))
        experiment.scheduler.load_state_dict(sheduler_dict)
        experiment.metrics.load_state_dict(checkpoint['metrics_state_dict'])
        experiment.system_metrics.load_state_dict(checkpoint['system_metrics_state_dict'])

        experiment._init_neptune(neptune)

        experiment.verbose = False
        for e in range(experiment.epoch + 1):
            system_metrics = experiment.system_metrics.metrics[e].avg
            experiment._log(f'\n{datetime.utcnow().isoformat()}\nlr: {system_metrics["lr"]}')
            # #
            metrics = experiment.metrics.train_metrics[e].avg
            experiment._log(f'Train epoch {e}, time: {system_metrics["train_epoch_time"]}s', **metrics)
            experiment._log_neptune('train', **metrics)
            # #
            metrics = experiment.metrics.valid_metrics[e].avg
            experiment._log(f'Valid epoch {e}, time: {system_metrics["valid_epoch_time"]}s', **metrics)
            experiment._log_neptune('valid', **metrics)
            # #
            experiment._log_neptune(**system_metrics)
            # #
            if experiment.low_memory:
                experiment.metrics.train_metrics[e].history = {}
                experiment.metrics.valid_metrics[e].history = {}

        experiment.verbose = experiment_state_dict['verbose']
        experiment.fit(train_loader, valid_loader, n_epochs - experiment.epoch - 1)

        return experiment
