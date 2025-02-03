import pickle
import torch
from torchtext.data.utils import get_tokenizer
from all_func import initialize_model

# Special tokens and language identifiers
TOKENS = {'UNK': 0, 'PAD': 1, 'SOS': 2, 'EOS': 3}
SRC_LANGUAGE, TRG_LANGUAGE = 'en', 'my'

def load_vocab(vocab_path='models/vocabs.pkl'):
    """Loads the vocabulary dictionary from a pickle file."""
    with open(vocab_path, 'rb') as f:
        return pickle.load(f)

def load_model(model_path='models/add_model_final.pt', device=None, vocab_transform=None):
    """
    Initializes the additive attention model and loads its saved state dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if vocab_transform is None:
        raise ValueError("vocab_transform must be provided.")
    
    model = initialize_model('additive', device, vocab_transform)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def tensor_transform(token_ids):
    """Wraps token IDs with <SOS> and <EOS> and converts them to a tensor."""
    return torch.tensor([TOKENS['SOS']] + token_ids + [TOKENS['EOS']])

def sequential_transforms(*transforms):
    """Chains multiple transformation functions together."""
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def translate_text(input_text, model, vocab_transform, device):
    """
    Translates input English text to Burmese using a greedy decoding approach.
    """
    tokenizer_src = get_tokenizer('spacy', language='en_core_web_sm')
    text_transform_src = sequential_transforms(
        tokenizer_src, 
        vocab_transform[SRC_LANGUAGE], 
        tensor_transform
    )
    
    src_tensor = text_transform_src(input_text.strip()).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    enc_src = model.encoder(src_tensor, src_mask)
    
    trg_indexes = [TOKENS['SOS']]
    max_len = 500
    
    for _ in range(max_len):
        trg_tensor = torch.tensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(-1)[:, -1].item()
        
        trg_indexes.append(pred_token)
        if pred_token == TOKENS['EOS']:
            break
    
    # Convert token indices to words
    itos = vocab_transform[TRG_LANGUAGE].get_itos()
    translation_tokens = [itos[idx] for idx in trg_indexes if idx not in {TOKENS['SOS'], TOKENS['EOS']}]
    
    return ' '.join(translation_tokens)
