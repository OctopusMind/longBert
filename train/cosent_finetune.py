import argparse
import math
import os

import torch
from dataset import CosentTrainDataset
from loguru import logger
from longbert_utils import get_sentence_embeddings
from text2vec.utils.stats_util import set_seed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def train(
    model,
    train_dataset: Dataset,
    output_dir: str,
    verbose: bool = True,
    batch_size: int = 4,
    num_epochs: int = 1,
    weight_decay: float = 0.01,
    seed: int = 42,
    warmup_ratio: float = 0.05,
    lr: float = 2e-5,
    eps: float = 1e-6,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    bf16: bool = False,
    encode_type="first_last_avg",
):
    """
    Trains the model on train_dataset.

    """
    os.makedirs(output_dir, exist_ok=True)
    device = None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    logger.debug("Use device: {}".format(device))
    model.to(device)
    set_seed(seed)

    num_devices = 1
    torch_type = torch.bfloat16 if bf16 else torch.float32
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )  # not shuffle

    total_steps = len(train_dataloader) * num_epochs // num_devices
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    warmup_steps = math.ceil(
        total_steps * warmup_ratio
    )  # by default 10% of _train data for warm-up
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Batch size = {batch_size}")
    logger.info(f"  Num steps = {total_steps}")
    logger.info(f"  Warmup-steps: {warmup_steps}")

    logger.info("  Training started")
    global_step = 0
    model.zero_grad()
    epoch_number = 0
    steps_trained_in_current_epoch = 0
    epochs_trained = 0

    def cosent_loss(y_true, y_pred):
        y_true = y_true[::2]
        norms = (y_pred**2).sum(axis=1, keepdims=True) ** 0.5
        y_pred = y_pred / norms
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
        y_pred = y_pred[:, None] - y_pred[None, :]
        y_true = y_true[:, None] < y_true[None, :]
        y_true = y_true.float()
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        y_pred = torch.cat((torch.tensor([0]).float().to(device), y_pred), dim=0)
        return torch.logsumexp(y_pred, dim=0)

    for _ in trange(int(num_epochs), desc="Epoch", disable=False, mininterval=0):
        model.train()
        if epochs_trained > 0:
            epochs_trained -= 1
            continue
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Running Epoch {epoch_number + 1} of {num_epochs}",
            disable=False,
            mininterval=0,
        )
        for step, batch in enumerate(batch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs, labels = batch
            labels = labels.to(device)

            inputs = {
                k: v.squeeze(1).to(device) for k, v in inputs.items() if v is not None
            }
            with torch.autocast(str(device), dtype=torch_type):
                output_embeddings = get_sentence_embeddings(
                    model, **inputs, encode_type=encode_type
                )

                loss = cosent_loss(labels, output_embeddings)
            current_loss = loss.item()
            if verbose:
                batch_iterator.set_description(
                    f"Epoch: {epoch_number + 1}/{num_epochs}, "
                    f"Batch:{step}/{len(train_dataloader) // num_devices}, Loss: {current_loss:9.4f}"
                )

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
        epoch_number += 1

    model.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")


def main():
    """
    >>> python cosent_finetune.py \
        --data_dir ../data/train_data.json \
        --output_dir ./outputs/my-model \
        --max_seq_length 1024 \
        --num_epochs 10 \
        --batch_size 64 \
        --learning_rate 2e-5
    """
    parser = argparse.ArgumentParser('Text Matching task')
    parser.add_argument('--data_dir', default='../data/train_data.json')
    parser.add_argument(
        '--output_dir',
        default='./outputs/my-model',
        type=str,
        help='Model output directory',
    )
    parser.add_argument(
        '--max_seq_length', default=128, type=int, help='Max sequence length'
    )
    parser.add_argument(
        '--num_epochs', default=3, type=int, help='Number of training epochs'
    )
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    parser.add_argument(
        '--learning_rate', default=2e-5, type=float, help='Learning rate'
    )
    args = parser.parse_args()
    logger.info(args)

    model_path = "OctopusMind/longbert-8k-zh"

    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, cache_dir="../model"
    )
    train_dataset = CosentTrainDataset(
        model.tokenizer, args.data_dir, max_len=args.max_seq_length
    )

    train(
        model,
        train_dataset,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
