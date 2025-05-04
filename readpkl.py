import pandas as pd
import matplotlib.pyplot as plt

file_path = "gpt_loss+lrs.pkl"

# Read the pickle file
data = pd.read_pickle(file_path)
print(type(data))
print("Keys in data:", data.keys())
for key in data.keys():
    print(key)
    # for entry in data[key]:
    #     plt.plot(data[key][entry])
    #     plt.xlabel("Iteration")
    #     plt.ylabel(f"{entry}")
    #     plt.title(f"{entry} vs Iteration of {key}")
    #     plt.grid()
    #     plt.show()
    plt.loglog(data[key]["Metrics/loss"])
    plt.xlabel("Iteration")
    plt.ylabel(f"Metrics/loss")
    plt.title(f"log-log Metrics/loss vs Iteration of {key}")
    plt.grid()
    plt.show()
