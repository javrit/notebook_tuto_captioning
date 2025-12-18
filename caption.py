import warnings


import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import imageio.v2 as imageio  # Utilisation de imageio.v2 pour éviter le warning de dépréciation
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Importez vos classes depuis models.py (assurez-vous que models.py se trouve dans le même dossier ou dans le PYTHONPATH)
from models import Encoder, Attention, DecoderWithAttention
torch.serialization.add_safe_globals([Encoder, DecoderWithAttention, Attention])

# PATCH: Monkey-patch pour torch.optim.Adam.__setstate__
import torch.optim
def patched_adam_setstate(self, state):
    if 'defaults' not in state:
        state['defaults'] = {'differentiable': False}
    super(torch.optim.Adam, self).__setstate__(state)
torch.optim.Adam.__setstate__ = patched_adam_setstate

# Analyse des arguments
parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
parser.add_argument('--img', '-i', help='path to image')
parser.add_argument('--model', '-m', help='path to model')
parser.add_argument('--word_map', '-wm', help='path to word map JSON')
parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
args = parser.parse_args()

# Charger le checkpoint en forçant weights_only=False (car weights_only=True ne fonctionne pas dans votre cas)
checkpoint = torch.load(args.model, map_location=str(device), weights_only=False)
if 'optimizer' in checkpoint:
    del checkpoint['optimizer']

decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Charger la word map (word2ix)
with open(args.word_map, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.
    
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: dictionary mapping words to indices
    :param beam_size: number of sequences to consider at each decode step
    :return: caption sequence (list of indices) and attention weights for visualization
    """
    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imageio.imread(image_path)
    if len(img.shape) == 2:
        # If grayscale, stack to get three channels
        img = np.stack([img] * 3, axis=-1)
    img = skimage.transform.resize(img, (256, 256), anti_aliasing=True)
    img = img.transpose(2, 0, 1)  # (3, 256, 256)
    # Pas besoin de diviser par 255, car resize normalise déjà dans [0,1]
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode the image
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding: (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    # Expand to beam size k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Initialize beams with <start> token
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # Beam search loop
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = top_k_words // vocab_size  # (k)
        next_word_inds = top_k_words % vocab_size   # (k)

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend([seqs[ind].tolist() for ind in complete_inds])
            complete_seqs_alpha.extend([seqs_alpha[ind].tolist() for ind in complete_inds])
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
            break
        step += 1

    if len(complete_seqs) == 0:
        complete_seqs = [seq.tolist() for seq in seqs]
        complete_seqs_scores = top_k_scores.squeeze(1).tolist()

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    return seq, alphas

def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with attention weights for each word.
    """
    image = Image.open(image_path)
    # Affichage de l'image d'origine (facultatif)
    plt.imshow(image)
    image = image.resize((14 * 24, 14 * 24), Image.LANCZOS)
    # Si vous ne voulez pas l'image de fond, commentez la ligne suivante :
    plt.imshow(image)

    words = [rev_word_map[ind] for ind in seq]
    n_rows = int(np.ceil(len(words) / 5.0))
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(n_rows, 5, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, (14 * 24, 14 * 24))
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig('caption.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Generate Caption')
    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location=str(device), weights_only=False)
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)

    print("Caption has been generated : check 'caption.png' to have a look at it")