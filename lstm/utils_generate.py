import numpy as np
import torch
import torch.nn as nn
import music21 as ms
from .utils_both_stacked import LSTMMusic, ltsm_gen, ltsm_gen_v2


# notes_encoded = load("chopin", "prelude", 15, time_step=0.25)
# Code used to train the network. Note that even if you run this you will not get the same network
# net, l, ll = train_lstm_loss_whole_seq(50, n_epochs=100, lr=0.01)
# To save the model
# torch.save(net.state_dict(), 'chopin_both_stacked.pkl')
def generate_music(trained_lstm, input_note, file_name):

    # Gets a NumPy array with all the frequency of the piano notes
    notes_freq = [ms.note.Note(note).pitch.frequency for note in ["A0", "A#0"]]
    s = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i in range(1, 8):
        notes_freq += [ms.note.Note(note + str(i)).pitch.frequency for note in s]
    notes_freq.append(ms.note.Note("C8").pitch.frequency)
    notes_freq = np.array(notes_freq)
    # Gets the note as a music21 object
    note_dict = {"do": "C", "re": "D", "mi": "E", "fa": "F", "sol": "G", "la": "A", "si": "B"}
    input_note_ms = ms.note.Note(note_dict[input_note])
    input_note_np = np.zeros((1, 89))
    input_note_np[0, :87] += (notes_freq == input_note_ms.pitch.frequency)*1
    input_note_np_left = np.zeros((1, 89))
    input_note_np = np.hstack((input_note_np_left, input_note_np))

    net = LSTMMusic(178, 178)
    net.load_state_dict(torch.load(trained_lstm, map_location=torch.device('cpu')))
    net.eval()

    ltsm_gen_v2(net, 50, file_name, torch.tensor(input_note_np), time_step=0.25, n_steps=600)
