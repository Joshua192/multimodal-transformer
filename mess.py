from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

# Add special tokens if needed
tokenizer.add_special_tokens({
    'pad_token': '<|endoftext|>',  # This is the default pad, but we make it explicit
    'bos_token': '<|startoftext|>',
    'eos_token': '<|endoftext|>',
    'unk_token': '<|endoftext|>'
})

MAX_LEN = 78
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
# DATA_PATH = Path("D:\huggingface_datasets\nlphuji___flickr30k\TEST\1.1.0\2b239befc81b6e3f035ce6bd52f5f4d60f5625f7")

class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform=image_transform, max_len=78):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = row['image']
        image = self.transform(image)
        
        caption = row['caption']

        encoding = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
from transformers import ViTModel

class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.linear = torch.nn.Linear(self.vit.config.hidden_size, 512)  # Project to decoder dimension
        self.norm = torch.nn.LayerNorm(512)

    def forward(self, x):
        with torch.no_grad():  # Freeze ViT for now
            outputs = self.vit(x)
            cls_embedding = outputs.last_hidden_state[:, 0]  # CLS token
        return self.norm(self.linear(cls_embedding)).unsqueeze(1)  # (B, 1, 512)

import torch.nn as nn
import torch.nn.functional as F

class CustomDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)  # Max sequence length
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True),
            num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        seq_len = tgt.size(1)
        positions = torch.arange(0, seq_len, device=tgt.device).unsqueeze(0)
        x = self.token_embedding(tgt) * (self.d_model ** 0.5) + self.pos_embedding(positions)
        x = self.dropout(x)
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(output)


def create_causal_mask(size):
    # Generates a lower triangular matrix for self-attention masking
    return torch.tril(torch.ones((size, size), dtype=torch.bool))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_encoder = ImageEncoder().to(device)
decoder = CustomDecoder(vocab_size=len(tokenizer)).to(device)
decoder.token_embedding.weight.data.normal_(mean=0.0, std=0.02)


def generate_caption(decoder, memory, tokenizer, device, max_len=78):
    decoder.eval()
    batch_size = memory.size(0)
    generated = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long).to(device)

    for _ in range(max_len):
        tgt_mask = create_causal_mask(generated.size(1)).to(device)
        outputs = decoder(generated, memory, tgt_mask=tgt_mask)
        next_token = outputs[:, -1, :].argmax(-1).unsqueeze(1)
        generated = torch.cat((generated, next_token), dim=1)

        # Stop if all sequences produced EOS
        if (next_token == tokenizer.eos_token_id).all():
            break

    captions = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
    return captions

def infer_caption(PIL_Image, image_encoder, decoder, tokenizer, device):

    image_tensor = image_transform(PIL_Image).unsqueeze(0).to(device)
    # Extract image features
    with torch.no_grad():
        memory = image_encoder(image_tensor)

    # Start with <BOS> token
    generated = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long).to(device)

    for _ in range(78):
        tgt_mask = create_causal_mask(generated.size(1)).to(device)
        output = decoder(generated, memory, tgt_mask=tgt_mask)
        next_token = output[:, -1, :].argmax(-1).unsqueeze(1)
        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    caption = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
    return caption
