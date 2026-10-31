from moabb.datasets import BNCI2014_001

dataset = BNCI2014_001()

print("Dataset:", dataset.code)
print("Subjects:", dataset.subject_list)

data = dataset.get_data(subjects=[1])

subject_data = data[1]

print("\nSessions:", list(subject_data.keys()))

for session, runs in subject_data.items():
    print(f"\nSession: {session}")

    for run, raw in runs.items():
        print(f"\nRun: {run}")
        print("Channels:", len(raw.ch_names))
        print("Sampling frequency:", raw.info["sfreq"])
        print("Duration:", round(raw.times[-1], 2), "seconds")
        print("Annotations:", raw.annotations.description[:10])