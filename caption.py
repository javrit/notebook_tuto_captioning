import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import imageio.v2 as imageio
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Import your classes (adjust the import if your classes are defined in a module)
# Add these classes to the safe globals list.
# Now load the checkpoint with weights_only=True.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import imageio  # replacing scipy.misc.imread
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import your classes from models.py
from models import Encoder, Attention, DecoderWithAttention
torch.serialization.add_safe_globals([Encoder, DecoderWithAttention, Attention])

# PATCH: Monkey-patch torch.optim.Adam.__setstate__ to add a default if missing.
import torch.optim
def patched_adam_setstate(self, state):
    if 'defaults' not in state:
        state['defaults'] = {'differentiable': False}
    super(torch.optim.Adam, self).__setstate__(state)
torch.optim.Adam.__setstate__ = patched_adam_setstate

# Now, parse arguments, etc.
parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
parser.add_argument('--img', '-i', help='path to image')
parser.add_argument('--model', '-m', help='path to model')
parser.add_argument('--word_map', '-wm', help='path to word map JSON')
parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
args = parser.parse_args()

# Load checkpoint with full state
checkpoint = torch.load(args.model, map_location=str(device))
if 'optimizer' in checkpoint:
    del checkpoint['optimizer']
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(args.word_map, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map (dictionary mapping words to indices)
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption sequence (list of indices), weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imageio.imread(image_path)
    if len(img.shape) == 2:
        # If grayscale, stack to get three channels
        img = np.stack([img]*3, axis=-1)
    # Resize using skimage.transform.resize; it returns a float array in [0,1]
    img = skimage.transform.resize(img, (256, 256), anti_aliasing=True)
    img = img.transpose(2, 0, 1)  # (3, 256, 256)
    # No need to divide by 255 because resize already normalizes to [0,1]
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

    # Expand to have a batch size equal to beam size k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words; start with <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; initially just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; initially zeros
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; initially ones
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is the number of active beams (<= k)
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add cumulative scores
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k beams have the same scores (only one <start> token)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (k)
        else:
            # Unroll and find top scores and their indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (k)

        # Convert unrolled indices to actual indices: prev beam index and next word index
        prev_word_inds = top_k_words // vocab_size  # (k)
        next_word_inds = top_k_words % vocab_size   # (k)

        # Add new words to sequences and alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (k, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (k, step+1, enc_image_size, enc_image_size)

        # Check which sequences have not reached <end>
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend([seqs[ind].tolist() for ind in complete_inds])
            complete_seqs_alpha.extend([seqs_alpha[ind].tolist() for ind in complete_inds])
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
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

    # Choose the best beam (highest cumulative log-probability)
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from the paper authors' repo.

    :param image_path: path to the image that has been captioned
    :param seq: caption (list of indices)
    :param alphas: attention weights
    :param rev_word_map: reverse word mapping, i.e., index-to-word dictionary
    :param smooth: whether to smooth the weights
    """
    image = Image.open(image_path)
    plt.imshow(image)

    image = image.resize((14 * 24, 14 * 24), Image.LANCZOS)
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
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    if 'optimizer' in checkpoint:
      del checkpoint['optimizer']
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
