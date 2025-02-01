#!/usr/bin/env python3

import json
import seaborn as sns
import matplotlib.pyplot as plt
import config

def get_bleu_scores(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convert BLEU score lists into PyTorch tensors
    token_level = data["token_level_bleu_list"]
    sentence_level = data["sentence_level_bleu_list"]

    return token_level, sentence_level


def get_distribution(model, token_level, sentence_level):
    # Convert tensors to Python lists for plotting
    data = {
        "Scores": token_level + sentence_level, 
        "Level": ["Token"] * len(token_level) + ["Sentence"] * len(sentence_level)
    }

    # Create a displot using our dictionary as the data source
    g = sns.displot(
        data=data, 
        x="Scores", 
        hue="Level", 
        kind="hist", 
        kde=False
    )
    
    # Optionally, you can label axes via the FacetGrid
    # (displot returns a FacetGrid, so we call set_axis_labels):
    g.set_axis_labels(model, "Count")  # x-label, y-label
    
    # Add a figure-level title. Note we adjust the `y` position slightly to avoid clipping:
    g.fig.suptitle("Distribution of BLEU Scores", y=1.03)
    
    # Force a layout that won’t clip titles/labels
    plt.tight_layout()
    
    # Save the figure with bbox_inches="tight" to ensure everything is in frame
    g.fig.savefig(f'{model} plot.png', dpi=300, bbox_inches='tight')
    
    # Show the plot interactively
    plt.show()

def plot(config):
    json_file_path = f"results.json"
    token_level, sentence_level = get_bleu_scores(json_file_path)
    get_distribution(config.MODEL, token_level, sentence_level)

plot(config)

