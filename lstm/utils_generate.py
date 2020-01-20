import numpy as np
import torch
import music21 as ms
from .utils_both_stacked import LSTMMusic, ltsm_gen


# notes_encoded = load("chopin", "prelude", 15, time_step=0.25)
# Code used to train the network. Note that even if you run this you will not get the same network
# net, l, ll = train_lstm_loss_whole_seq(50, n_epochs=100, lr=0.01)
# To save the model
# torch.save(net.state_dict(), 'chopin_both_stacked.pkl')
def generate_music(trained_lstm, input_notes_seq, file_name,
                   n_steps=300, hold_thres=[0.7, 0.7], note_thres=[0.9, 0.9],
                   seq_len=50, tempo=74, both_hands=True):

    # Gets a NumPy array with all the frequency of the piano notes
    notes_freq = [ms.note.Note(note).pitch.frequency for note in ["A0", "A#0"]]
    s = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i in range(1, 8):
        notes_freq += [ms.note.Note(note + str(i)).pitch.frequency for note in s]
    notes_freq.append(ms.note.Note("C8").pitch.frequency)
    notes_freq = np.array(notes_freq)

    # TODO: Include support for chords (if copy-pasting from somewhere)
    input_notes = input_notes_seq.split(" ")
    input_notes_np = np.zeros((len(input_notes), 89))
    for i, input_note in enumerate(input_notes):
        # Gets the note as a music21 object
        if input_note:
            # input_note = input_note.replace("3", "1").replace("4", "8")
            input_note_ms = ms.note.Note(input_note, duration=ms.duration.Duration(0.25))
            input_notes_np[i, :87] += (notes_freq == input_note_ms.pitch.frequency)*1

    if both_hands:
        input_notes_np_left = np.zeros((len(input_notes), 89))
        input_notes_np = np.hstack((input_notes_np_left, input_notes_np))
        hidden_size = 178
    else:
        hidden_size = 89

    net = LSTMMusic(hidden_size, hidden_size)
    net.load_state_dict(torch.load(trained_lstm, map_location=torch.device('cpu')))
    net.eval()

    ltsm_gen(net, file_name, torch.tensor(input_notes_np), time_step=0.25, n_steps=n_steps,
             hold_thres=hold_thres, note_thres=note_thres, seq_len=seq_len, tempo=tempo)
