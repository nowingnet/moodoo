import matplotlib.pyplot as plt
import numpy as np
import torch
from helper.nerf_helpers import meshgrid_xy
import torchvision

#CELL#16
def save_plt_image(im1, outname):
    fig = plt.figure()
    fig.set_size_inches((6.4,6.4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #plt.set_cmap('jet')
    ax.imshow(im1, aspect='equal')
    plt.savefig(outname, dpi=80)
    plt.close(fig)


def normal_map_from_depth_map(depthmap):
    h, w = np.shape(depthmap)
    normals = np.zeros((h, w, 3))
    phong = np.zeros((h, w, 3))
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            dzdx = (float((depthmap[x + 1, y])) - float((depthmap[x - 1, y]))) / 2.0
            dzdy = (float((depthmap[x, y + 1])) - float((depthmap[x, y - 1]))) / 2.0

            n = np.array([-dzdx, -dzdy, 0.005])

            n = n * 1/np.linalg.norm(n)
            dir = np.array([x,y,1.0])
            dir = dir *1/np.linalg.norm(dir)

            normals[x, y] = (n*0.5 + 0.5)
            phong[x, y] = np.dot(dir,n)*0.5+0.5

    normals *= 255
    normals = normals.astype('uint8')
    #plt.imshow(depthmap, cmap='gray')
    #plt.show()
    plt.imshow(normals)
    plt.show()
    plt.imshow(phong)
    plt.show()
    print('a')
    return normals

def torch_normal_map(depthmap,focal,weights=None,clean=True, central_difference=False):
    W,H = depthmap.shape
    #normals = torch.zeros((H,W,3), device=depthmap.device)
    cx = focal[2]*W
    cy = focal[3]*H
    fx = focal[0]
    fy = focal[1]
    ii, jj = meshgrid_xy(torch.arange(W, device=depthmap.device),
                         torch.arange(H, device=depthmap.device))
    points = torch.stack(
        [
            ((ii - cx) * depthmap) / fx,
            -((jj - cy) * depthmap) / fy,
            depthmap,
        ],
        dim=-1)
    difference = 2 if central_difference else 1
    dx = (points[difference:,:,:] - points[:-difference,:,:])
    dy = (points[:,difference:,:] - points[:,:-difference,:])
    normals = torch.cross(dy[:-difference,:,:],dx[:,:-difference,:],2)
    normalize_factor = torch.sqrt(torch.sum(normals*normals,2))
    normals[:,:,0]  /= normalize_factor
    normals[:,:,1]  /= normalize_factor
    normals[:,:,2]  /= normalize_factor
    normals = normals * 0.5 +0.5

    if clean and weights is not None: # Use volumetric rendering weights to clean up the normal map
        mask = weights.repeat(3,1,1).permute(1,2,0)
        mask = mask[:-difference,:-difference]
        where = torch.where(mask > 0.22)
        normals[where] = 1.0
        normals = (1-mask)*normals + (mask)*torch.ones_like(normals)
    normals *= 255
    #plt.imshow(normals.cpu().numpy().astype('uint8'))
    #plt.show()
    return normals

def vis(tensor):
    plt.imshow((tensor*255).cpu().numpy().astype('uint8'))
    plt.show()
def normal_map_from_depth_map_backproject(depthmap):
    h, w = np.shape(depthmap)
    normals = np.zeros((h, w, 3))
    phong = np.zeros((h, w, 3))
    cx = cy = h//2
    fx=fy=500
    fx = fy = 1150
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            #dzdx = (float((depthmap[x + 1, y])) - float((depthmap[x - 1, y]))) / 2.0
            #dzdy = (float((depthmap[x, y + 1])) - float((depthmap[x, y - 1]))) / 2.0

            p = np.array([(x*depthmap[x,y]-cx)/fx, (y*depthmap[x,y]-cy)/fy, depthmap[x,y]])
            py = np.array([(x*depthmap[x,y+1]-cx)/fx, ((y+1)*depthmap[x,y+1]-cy)/fy, depthmap[x,y+1]])
            px = np.array([((x+1)*depthmap[x+1,y]-cx)/fx, (y*depthmap[x+1,y]-cy)/fy, depthmap[x+1,y]])

            #n = np.array([-dzdx, -dzdy, 0.005])
            n = np.cross(px-p, py-p)
            n = n * 1/np.linalg.norm(n)
            dir = p#np.array([x,y,1.0])
            dir = dir *1/np.linalg.norm(dir)

            normals[x, y] = (n*0.5 + 0.5)
            phong[x, y] = np.dot(dir,n)*0.5+0.5

    normals *= 255
    normals = normals.astype('uint8')
    #plt.imshow(depthmap, cmap='gray')
    #plt.show()
    #plt.imshow(normals)
    #plt.show()
    #plt.imshow(phong)
    #plt.show()
    #print('a')
    return normals

