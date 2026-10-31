import mne
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001

dataset = BNCI2014_001()
data = dataset.get_data(subjects=[1])

subject = data[1]
session = list(subject.keys())[0]
run = list(subject[session].keys())[0]

raw = subject[session][run].copy()

raw.filter(8, 30, fir_design="firwin")

events, event_id = mne.events_from_annotations(raw)

selected_events = {
    label: code
    for label, code in event_id.items()
    if label in ["left_hand", "right_hand", "feet", "tongue"]
}

epochs = mne.Epochs(
    raw,
    events,
    event_id=selected_events,
    tmin=0.5,
    tmax=4.0,
    baseline=None,
    preload=True,
    picks="eeg",
    verbose=False
)

print("Epoch shape:", epochs.get_data().shape)
print("Number of trials:", len(epochs))
print("Sampling frequency:", epochs.info["sfreq"])

epochs.compute_psd(fmin=1, fmax=40).plot()
plt.show()

epochs.save("data/processed/subject_01-epo.fif", overwrite=True)