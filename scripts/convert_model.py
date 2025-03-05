import sys
sys.path.insert(0, '.')

from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn

from configs import BaseConfig, CONFIGS
from src.model import get_ocr_model, RecognitionModel
from src.experiment import OCRExperiment
from src.ctc_labeling import CTCLabeling


def main():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--from', type=str)
    parser.add_argument('--to', type=str)
    parser.add_argument('--output', type=Path)

    args = parser.parse_args()
    args.checkpoint = args.checkpoint.absolute()
    args.output = args.output.absolute()

    placeholder_config_args = {
        'data_dir': '',
        'image_w': 0,
        'image_h': 0
    }

    from_config: BaseConfig = CONFIGS[getattr(args, 'from')](**placeholder_config_args)
    to_config: BaseConfig = CONFIGS[args.to](**placeholder_config_args)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    from_model: RecognitionModel = get_ocr_model(from_config, pretrained=True)
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    optimizer = torch.optim.AdamW(from_model.parameters(), **from_config['optimizer']['params'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=1,
        steps_per_epoch=1,
        **from_config['scheduler']['params'],
    )
    ctc_labeling = CTCLabeling(from_config)

    experiment = OCRExperiment.from_checkpoint(
        checkpoint_path=args.checkpoint,
        train_loader=None,
        valid_loader=None,
        n_epochs=0,
        model=from_model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        ctc_labeling=ctc_labeling,
    )
    
    model: RecognitionModel = experiment.model
    old_layer: nn.Linear = model.classifier[3]
    print(old_layer)

    new_layer = nn.Linear(
        in_features=old_layer.in_features,
        out_features=len(to_config.blank + to_config.chars)
    )
    new_layer.weight.data[:old_layer.out_features] = old_layer.weight.data[:]
    new_layer.bias.data[:old_layer.out_features]   = old_layer.bias.data[:]

    nn.init.xavier_uniform_(new_layer.weight.data[old_layer.out_features:])
    nn.init.zeros_(new_layer.bias.data[old_layer.out_features:])

    setattr(experiment.model.classifier, '3', new_layer)

    experiment.optimizer = torch.optim.AdamW(experiment.model.parameters(), **to_config['optimizer']['params'])
    experiment.save(args.output)

if __name__ == '__main__':
    main()
