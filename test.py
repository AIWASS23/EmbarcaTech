import torch
from torch.utils.data import Dataset
from src.vlm.captioning.trainer_pipeline.imageprocessor import ImageProcessor
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.vlm.captioning.trainer_pipeline.captiondataset.datasetprocessor import ImageCaptioningDatasetProcessor
from src.vlm.captioning.trainer_pipeline.captiondataset.datasettokenizer import ImageCaptioningDatasetTokenizer
from src.pdi.data_augmentation import DataAugmentation



class ImageCaptioningDatasetTokenizer(Dataset):
    def __init__(self, df, image_folders, processor, tokenizer, column, max_length=50):
        self.df = df
        self.image_folders = image_folders
        self.processor = processor
        self.tokenizer = tokenizer
        self.column = column
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = ImageProcessor.get_image_path(row['id_image'], self.image_folders)
        caption = row[self.column]

        if image_path is None:
            print(f"Warning: Image not found for id_image {row['id_image']}")
            return None

        image = ImageProcessor.load_image(image_path)

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        inputs = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = inputs.input_ids.squeeze()
        return {"pixel_values": pixel_values.squeeze(), "input_ids": input_ids}

    @staticmethod
    def collate_fn_tokenizer(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        return {"pixel_values": pixel_values, "input_ids": input_ids}


class ImageCaptioningDatasetProcessor(Dataset):
    def __init__(self, df, image_folder, processor, column):
        self.df = df
        self.image_folder = image_folder
        self.processor = processor
        self.column = column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      
        row = self.df.iloc[idx]
        image_path = ImageProcessor.get_image_path(row['id_image'], self.image_folder)
        caption = row[self.column]

        if image_path is None:
            print(f"Warning: Image not found for id_image {row['id_image']}")
            return None

        image = ImageProcessor.load_image(image_path)
        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = caption
        return encoding

    @staticmethod
    def collate_fn_processor(batch, processor):
        batch = [item for item in batch if item is not None]
        processed_batch = {}

        if len(batch) == 0:
            return None

        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch
    

class Trainer:
    
    def __init__(
            self,
            model_manager,
            processor,
            df,
            image_folder,
            batch_size,
            lr,
            num_epochs,
            column,
            optimizer_cls,
            loss_fn=None,
            scheduler_cls=None,
            scheduler_params=None,
            optimizer_params=None,
            tokenizer=None,
            augmentation_steps=None):

        self.model_manager = model_manager
        self.processor = processor
        self.tokenizer = tokenizer
        self.df = df
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.column = column

        self.augmentation_steps = augmentation_steps
        self.model = self.model_manager.get_model()
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, **optimizer_params)

        if scheduler_cls is not None:
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None

        self.loss_fn = loss_fn
        self.augmenter = DataAugmentation(augmentation_steps) if augmentation_steps else None

        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        self.writer = SummaryWriter(log_dir="./tensorboard_logs")

    def calculate_accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        accuracy = correct.sum() / len(correct)
        return accuracy.item()

    def apply_augmentation(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if self.augmenter:
            augmented_images = self.augmenter.apply_steps([image])
            return augmented_images
        return [image]

    def prepare_loss_input(self, logits, input_ids):
        if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
            logits = logits[:, :input_ids.size(1), :]  # logits e input_ids alinhados
            logits = logits.reshape(-1, logits.size(-1))
            input_ids = input_ids.view(-1)
            return logits, input_ids

        elif isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            targets = torch.nn.functional.one_hot(input_ids, num_classes=logits.size(-1)).float()
            return logits.view(-1, logits.size(-1)), targets.view(-1, logits.size(-1))

        elif isinstance(self.loss_fn, torch.nn.L1Loss) or isinstance(self.loss_fn, torch.nn.MSELoss):
            targets = torch.nn.functional.one_hot(input_ids, num_classes=logits.size(-1)).float()
            return logits.view(-1), targets.view(-1)

        elif isinstance(self.loss_fn, torch.nn.KLDivLoss):
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            targets = torch.nn.functional.one_hot(input_ids, num_classes=logits.size(-1)).float()
            targets = targets / targets.sum(dim=-1, keepdim=True)
            return logits, targets

        elif isinstance(self.loss_fn, torch.nn.CosineEmbeddingLoss):
            targets = (input_ids.float() == input_ids.float().mean()).float() * 2 - 1
            return logits.view(-1), logits.view(-1), targets.view(-1)

        elif isinstance(self.loss_fn, torch.nn.NLLLoss):
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            return logits.view(-1, logits.size(-1)), input_ids.view(-1)

        else:
            return None

    def train(self):
        if self.tokenizer is not None:
            train_dataset = ImageCaptioningDatasetTokenizer(self.df, self.image_folder, self.processor, self.tokenizer, self.column)
            collate_fn = lambda batch: ImageCaptioningDatasetTokenizer.collate_fn_tokenizer(batch)

        else:
            train_dataset = ImageCaptioningDatasetProcessor(self.df, self.image_folder, self.processor, self.column)
            collate_fn = lambda batch: ImageCaptioningDatasetProcessor.collate_fn_processor(batch, self.processor)

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(device)
        self.model.train()

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            total_loss = 0
            total_accuracy = 0

            for idx, batch in enumerate(train_dataloader):
                augmented_images = []
                for image in batch["pixel_values"]:
                    augmented_images.extend(self.apply_augmentation(image))

                print(f"Total images in batch after enlargement: {len(augmented_images)}")

                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device).to(torch.float32)

                self.optimizer.zero_grad()

                if self.tokenizer is None:
                    outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
                else:
                    outputs = self.model(pixel_values=pixel_values, labels=input_ids)

                logits = outputs.logits
                prepared_inputs = self.prepare_loss_input(logits, input_ids)

                if self.loss_fn is None or prepared_inputs is None or len(prepared_inputs) < 2:
                    loss = outputs.loss.mean()
                else:
                    loss = self.loss_fn(*prepared_inputs)

                total_loss += loss.mean().item()

                batch_accuracy = self.calculate_accuracy(logits, input_ids)
                total_accuracy += batch_accuracy

                print(f"Batch {idx + 1}, Loss: {loss.mean().item()}, Accuracy: {batch_accuracy:.4f}")

                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()

            if self.scheduler is not None:
                self.scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            avg_train_accuracy = total_accuracy / len(train_dataloader)

            self.metrics['train_loss'].append(avg_train_loss)
            self.metrics['train_accuracy'].append(avg_train_accuracy)

            avg_val_loss, avg_val_accuracy = self.evaluate(self.df, self.image_folder, self.column)
            self.metrics['val_loss'].append(avg_val_loss)
            self.metrics['val_accuracy'].append(avg_val_accuracy)

            output_dir = "./fine-tuning"
            self.model_manager.save_model(output_dir)
            print("Modelo e processador salvos com sucesso!")

        self.save_metrics()
        self.writer.close()  # tensorboard --logdir=./tensorboard_logs


    def evaluate(self, df_eval, image_folder, column):
        if self.tokenizer is not None:
            eval_dataset = ImageCaptioningDatasetTokenizer(df_eval, image_folder, self.processor, self.tokenizer, column)
            collate_fn = lambda batch: ImageCaptioningDatasetTokenizer.collate_fn_tokenizer(batch)
        else:
            eval_dataset = ImageCaptioningDatasetProcessor(df_eval, image_folder, self.processor, column)
            collate_fn = lambda batch: ImageCaptioningDatasetProcessor.collate_fn_processor(batch, self.processor)

        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

        total_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device).to(torch.float32)

                if self.tokenizer is None:
                    outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
                else:
                    outputs = self.model(pixel_values=pixel_values, labels=input_ids)

                logits = outputs.logits
                prepared_inputs = self.prepare_loss_input(logits, input_ids)

                if self.loss_fn is None or prepared_inputs is None or len(prepared_inputs) < 2:
                    loss = outputs.loss.mean()
                else:
                    loss = self.loss_fn(*prepared_inputs)

                total_loss += loss.mean().item()

                batch_accuracy = self.calculate_accuracy(logits, input_ids)
                total_accuracy += batch_accuracy

        avg_loss = total_loss / len(eval_dataloader)
        avg_accuracy = total_accuracy / len(eval_dataloader)

        return avg_loss, avg_accuracy

    def save_metrics(self, file_path="metrics.json"):
        with open(file_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)