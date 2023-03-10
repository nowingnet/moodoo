import numpy as np
import torchvision
import matplotlib.pyplot as plt

#CELL#14 
def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img 


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)

def error_image(im1, im2):
    fig = plt.figure()
    diff = (im1 - im2)
    #gt_vs_theirs[total_mask, :] = 0
    #print("theirs ", np.sqrt(np.sum(np.square(gt_vs_theirs))), np.mean(np.square(gt_vs_theirs)))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    # Then we disable our xaxis and yaxis completely. If we just say plt.axis('off'),
    # they are still used in the computation of the image padding.
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Even though our axes (plot region) are set to cover the whole image with [0,0,1,1],
    # by default they leave padding between the plotted data and the frame. We use tigher=True
    # to make sure the data gets scaled to the full extents of the axes.
    plt.autoscale(tight=True)
    plt.imshow(np.linalg.norm(diff, axis=2), cmap='jet')
    #ax.plt.axes('off')



    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax.set_axis_off()
    #plt.show()
    return fig
