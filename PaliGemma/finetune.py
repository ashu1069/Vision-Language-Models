"""
Fine-tuning script for PaliGemma model.
Supports training on image-text pairs with various training configurations.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from PIL import Image
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fire
from tqdm import tqdm
import logging

from paligemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from processing import PaliGemmaProcessor
from utils import load_hf_model
from gemma import KVCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaliGemmaDataset(Dataset):
    """
    Dataset for PaliGemma fine-tuning.
    Expects a JSON file with format:
    [
        {
            "image_path": "path/to/image.jpg",
            "text": "caption or instruction"
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data_file: str,
        processor: PaliGemmaProcessor,
        max_length: int = 512,
    ):
        self.processor = processor
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        text = item["text"]
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (224, 224), color="white")
        
        # Process with processor (no padding - handled in collate_fn)
        model_inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=True,
        )
        
        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "pixel_values": model_inputs["pixel_values"].squeeze(0),
        }


def create_collate_fn(pad_token_id: int = 0):
    """
    Create a collate function with the specified pad_token_id.
    Since the model requires attention_mask to be all ones (no padding support),
    we pad sequences to the same length but set attention_mask to all ones.
    Padding tokens will be masked out in the loss computation.
    """
    def collate_fn(batch):
        # Find max length in batch
        max_len = max(item["input_ids"].shape[0] for item in batch)
        
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        
        for item in batch:
            seq_len = item["input_ids"].shape[0]
            
            # Pad input_ids if necessary
            if seq_len < max_len:
                pad_len = max_len - seq_len
                padded_input_ids = torch.cat([
                    item["input_ids"],
                    torch.full((pad_len,), pad_token_id, dtype=item["input_ids"].dtype)
                ])
            else:
                padded_input_ids = item["input_ids"]
            
            input_ids_list.append(padded_input_ids)
            
            # Create attention mask - all ones as required by model
            # Note: The model will process padding tokens, but we mask them in loss
            attention_mask_list.append(torch.ones(max_len, dtype=torch.long))
            pixel_values_list.append(item["pixel_values"])
        
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        pixel_values = torch.stack(pixel_values_list)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
    
    return collate_fn


def create_labels(input_ids: torch.Tensor, pad_token_id: int = 0, ignore_index: int = -100) -> torch.Tensor:
    """
    Create labels for training. Shift input_ids by one position for next token prediction.
    Also mask out image tokens and padding tokens.
    """
    batch_size, seq_len = input_ids.shape
    labels = input_ids.clone()
    
    # Shift labels by one position for next token prediction
    labels[:, :-1] = input_ids[:, 1:].clone()
    labels[:, -1] = ignore_index
    
    # Mask out padding tokens
    labels[input_ids == pad_token_id] = ignore_index
    
    return labels


def compute_loss(
    model: PaliGemmaForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute the training loss.
    Only compute loss on text tokens, not on image tokens or padding.
    """
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=None,  # No KV cache during training
    )
    
    logits = outputs["logits"]  # [batch_size, seq_len, vocab_size]
    
    # Reshape for loss calculation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten for cross entropy
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Create loss function
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(shift_logits, shift_labels)
    
    return loss


def train_epoch(
    model: PaliGemmaForConditionalGeneration,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pad_token_id: int = 0,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
    ignore_index: int = -100,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_steps = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        
        # Create labels (mask padding tokens)
        labels = create_labels(input_ids, pad_token_id=pad_token_id, ignore_index=ignore_index)
        labels = labels.to(device)
        
        # Forward pass with optional mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = compute_loss(
                    model, input_ids, pixel_values, attention_mask, labels, ignore_index
                )
                loss = loss / gradient_accumulation_steps
        else:
            loss = compute_loss(
                model, input_ids, pixel_values, attention_mask, labels, ignore_index
            )
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_steps += 1
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss


def evaluate(
    model: PaliGemmaForConditionalGeneration,
    dataloader: DataLoader,
    device: str,
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> float:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    num_steps = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            # Create labels
            labels = create_labels(input_ids, pad_token_id=pad_token_id, ignore_index=ignore_index)
            labels = labels.to(device)
            
            # Forward pass
            loss = compute_loss(
                model, input_ids, pixel_values, attention_mask, labels, ignore_index
            )
            
            total_loss += loss.item()
            num_steps += 1
            
            progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss


def save_checkpoint(
    model: PaliGemmaForConditionalGeneration,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(
    model: PaliGemmaForConditionalGeneration,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: str,
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, loss {loss})")
    
    return epoch, loss


def main(
    # Model and data paths
    model_path: str,
    train_data_file: str,
    val_data_file: Optional[str] = None,
    
    # Training hyperparameters
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    
    # Data parameters
    max_length: int = 512,
    num_workers: int = 4,
    
    # Training options
    use_amp: bool = True,
    only_cpu: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    
    # Output
    output_dir: str = "./checkpoints",
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 10,
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine",  # "cosine" or "linear"
):
    """
    Fine-tune PaliGemma model on image-text pairs.
    
    Args:
        model_path: Path to the pre-trained PaliGemma model
        train_data_file: JSON file with training data (list of {"image_path": ..., "text": ...})
        val_data_file: Optional JSON file with validation data
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        max_length: Maximum sequence length
        num_workers: Number of data loader workers
        use_amp: Use automatic mixed precision training
        only_cpu: Force CPU usage
        resume_from_checkpoint: Path to checkpoint to resume from
        output_dir: Directory to save checkpoints
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        lr_scheduler_type: Type of learning rate scheduler ("cosine" or "linear")
    """
    
    # Setup device
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backend.mps.is_available():
            device = "mps"
            logger.info("Using MPS device")
    
    logger.info(f"Device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}...")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device)
    
    # Setup processor
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    
    # Get pad_token_id from model config
    pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    
    # Create collate function with pad_token_id
    collate_fn = create_collate_fn(pad_token_id=pad_token_id)
    
    # Setup datasets
    train_dataset = PaliGemmaDataset(train_data_file, processor, max_length=max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )
    
    val_dataloader = None
    if val_data_file:
        val_dataset = PaliGemmaDataset(val_data_file, processor, max_length=max_length)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True if device == "cuda" else False,
        )
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Calculate total training steps
    num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    
    # Setup learning rate scheduler
    if lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    elif lr_scheduler_type == "linear":
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps)
    else:
        scheduler = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    
    if resume_from_checkpoint:
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from_checkpoint, device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Total training steps: {num_training_steps}")
    
    global_step = 0
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            device,
            pad_token_id=pad_token_id,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
            ignore_index=model.config.ignore_index,
        )
        
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Evaluate
        if val_dataloader:
            val_loss = evaluate(
                model,
                val_dataloader,
                device,
                pad_token_id=pad_token_id,
                ignore_index=model.config.ignore_index,
            )
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # Save checkpoint
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                val_loss,
                output_dir,
                is_best=is_best,
            )
        else:
            # Save checkpoint without validation
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                train_loss,
                output_dir,
                is_best=False,
            )
        
        global_step += len(train_dataloader)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    fire.Fire(main)

