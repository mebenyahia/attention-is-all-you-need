import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class Visualizer:
    def __init__(self, model, patch_size):
        self.model = model
        self.patch_size = patch_size

    def visualize_predict(self, img):
        attention = self.visualize_attention(img)
        self.plot_attention(img, attention)

    def visualize_attention(self, img):
        # make the image divisible by the patch size

        w_featmap = img.shape[-2] // self.patch_size
        h_featmap = img.shape[-1] // self.patch_size

        attentions = self.model.get_last_selfattention(img)

        # print(attentions.shape)
        # prints: torch.Size([1, 4, 65, 65]) for IMAGE_SIZE 32x32 and PATCH_SIZE 4x4

        nh = attentions.shape[1]  # number of head

        # keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0), scale_factor=self.patch_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )

        return attentions

    def plot_attention(self, img, attention):
        n_heads = attention.shape[0]

        fig = plt.figure(constrained_layout=True, figsize=(12, 9))
        subfigs = fig.subfigures(2, 1, wspace=0.01, hspace=0.01)

        top_axis = subfigs[0].subplots(1, 3, width_ratios=[4, 4, 0.2])
        bottom_axis = subfigs[1].subplots(1, n_heads)

        text = ["Original Image", "Head Max"]
        img = img.cpu().numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))
        for i, fig in enumerate([img, np.max(attention, 0)]):
            comap = top_axis[i].imshow(fig, cmap="viridis")
            top_axis[i].set_title(text[i])
            top_axis[i].axis("off")
            if i == 1:
                plt.colorbar(comap, cax=top_axis[2])
                top_axis[2].set_title("Attention Level")

        for i in range(n_heads):
            bottom_axis[i].imshow(attention[i], cmap="viridis")
            bottom_axis[i].set_title(f"Head n: {i+1}")
            bottom_axis[i].axis("off")

        plt.show()
