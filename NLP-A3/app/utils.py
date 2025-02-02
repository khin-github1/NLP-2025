# util_funs.py
import pickle
import torch
from torchtext.data.utils import get_tokenizer
from all_func import initialize_model

# Special tokens and languages (should match training)
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'my'

def load_vocab(vocab_path='models/vocabs.pkl'):
    """
    Loads the vocabulary (a dict with keys 'en' and 'my') from a pickle file.
    """
    with open(vocab_path, 'rb') as f:
        vocab_transform = pickle.load(f)
    return vocab_transform

def load_model(model_path='models/add_model_final.pt', device=None, vocab_transform=None):
    """
    Initializes the additive attention model using the parameters from your file,
    then loads the saved state dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if vocab_transform is None:
        raise ValueError("vocab_transform must be provided.")
    # Initialize model with additive attention
    model = initialize_model('additive', device, vocab_transform)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def tensor_transform(token_ids):
    """
    Adds <sos> and <eos> tokens to a list of token ids and converts to a tensor.
    """
    import torch
    return torch.cat((
        torch.tensor([SOS_IDX]),
        torch.tensor(token_ids),
        torch.tensor([EOS_IDX])
    ))

def sequential_transforms(*transforms):
    """
    Composes multiple transforms into one.
    """
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def translate_text(input_text, model, vocab_transform, device):
    """
    Given English input text, tokenizes and numericalizes it, passes it through the model,
    and returns a Burmese translation (using a greedy decoding approach).
    """
    # For English, we use spaCy tokenizer
    tokenizer_src = get_tokenizer('spacy', language='en_core_web_sm')
    text_transform_src = sequential_transforms(
        tokenizer_src,
        vocab_transform[SRC_LANGUAGE],
        tensor_transform
    )
    src_tensor = text_transform_src(input_text.strip())
    src_tensor = src_tensor.unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [SOS_IDX]
    max_len = 500  # maximum translation length
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(-1)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == EOS_IDX:
            break
    # Convert token indices to words using the target vocab's itos mapping
    itos = vocab_transform[TRG_LANGUAGE].get_itos()
    translation_tokens = [itos[idx] for idx in trg_indexes]
    if translation_tokens[0] == '<sos>':
        translation_tokens = translation_tokens[1:]
    if translation_tokens[-1] == '<eos>':
        translation_tokens = translation_tokens[:-1]
    translation_text = ' '.join(translation_tokens)
    return translation_text