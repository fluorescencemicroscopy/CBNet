import os, sys, csv, time
import torch
import numpy as np
import skimage.io as skio
import skimage
import imageio

import retinanet.model as retinanet_model
from retinanet.dataloader import Resizer
from conv_xyz import ConvXYZ
from conv_gamma import ConvGamma

def resize(image, min_side = 608, max_side = 1024):
    try:
        rows, cols, cns = image.shape
    except:
        raise ValueError('The shape of image should have three components rows, cols, channels.')

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    # scale = min_side / smallest_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    # if largest_side * scale > max_side:
    #     scale = max_side / largest_side

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))

    rows, cols, cns = image.shape

    pad_w = 32 - rows%32
    pad_h = 32 - cols%32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)

    return new_image, scale

def analyze(
    model_retinanet,
    model_convxyz,
    model_gamma,
    images,
    prob = 0.8,
    amplitude = 4095.,
    min_side = 512,
    max_side = 1024,
    n_rows_convxyz = 32, #model_convxyz.nrows
    n_cols_convxyz = 32, #model_convxyz.ncols
    pixel_size = 0.11,
    outputdirectory = "outputs",
    resultname = "result.csv"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_retinanet.to(device)
    model_retinanet.eval()
    model_retinanet.freeze_bn()

    model_convxyz.to(device)
    model_convxyz.eval()

    model_gamma.to(device)
    model_gamma.eval()

    nframes, nrows, ncols = images.shape

    if not os.path.isdir(outputdirectory):
        os.mkdir(outputdirectory)

    f = open(os.path.join(outputdirectory, resultname), 'w', newline='')
    csvw_temp = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
    csvw_temp.writerow(['frame','x (px)','y (px)','z (um)','intensity', 'est. unc. int.'])

    def get_(x_c, y_c, nrows, ncols):
        x_i = int(round(x_c)) - n_cols_convxyz//2
        x_f = x_i + n_cols_convxyz
        if x_i < 0:
            x_i = 0
            x_f = n_cols_convxyz

        if x_f >= ncols:
            x_f = ncols
            x_i = x_f - n_cols_convxyz

        y_i = int(round(y_c)) - n_rows_convxyz//2
        y_f = y_i + n_rows_convxyz
        if y_i < 0:
            y_i = 0
            y_f = n_rows_convxyz

        if y_f >= ncols:
            y_f = ncols
            y_i = y_f - n_rows_convxyz

        return x_i, y_i, x_f, y_f

    results = []
    start = time.time()
    for i, img in enumerate(images):
        start0 = time.time()
        # preprocess image
        image0_c = np.expand_dims(img.astype(np.float32), axis = 2)
        image, scale = resize(image0_c, min_side = min_side, max_side = max_side)
        image /= amplitude # normalize pixel values.
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
    
        # process image
        image = torch.from_numpy(image).to(device).float()
        scores, classifications, transformed_anchors = model_retinanet(image)

        result_retinanet = []
        images_t = []
        for score, label, bbox in zip(scores.cpu(), classifications.cpu(), transformed_anchors.cpu()):
            if score < prob:
                break
            bbox /= scale
            tmp0 = [i, *(bbox.detach().numpy()), score.detach().item(), label.detach().item()]
            result_retinanet.append(tmp0)

        if result_retinanet:
            cropped_indices = []
            for res_t in result_retinanet:
                x0, y0, x1, y1 = res_t[1:5]
                x0, y0, x1, y1 = get_((x0 + x1)/2, (y0 + y1)/2, nrows, ncols)
                cropped_indices.append([res_t[0], x0, y0])
                x = image0_c[y0:y1, x0:x1]
                x = np.transpose(x, (2, 0, 1))
                images_t.append(x)
            images_t = np.asarray(images_t)
            nim_t = images_t[:]/4095
            nim_t = nim_t.astype('float32')
            nniimm = torch.from_numpy(nim_t).to(device).float()
            nphout = model_gamma(nniimm)
            nph_t = nphout.cpu().detach().numpy()
            nph_t[:,1] = np.exp(nph_t[:,1])
            nph_t *= 50000
            images_t /= amplitude
            x = torch.from_numpy(images_t).float().to(device)
            pos = model_convxyz(x)
            pos_t = pos.detach().cpu().numpy()
            pos_t[:, 0] *= n_cols_convxyz
            pos_t[:, 1] *= n_rows_convxyz
            pos_t[:, 2] *= 2.585
            z = np.hstack([np.asarray(cropped_indices), pos_t, nph_t])
            newz = np.zeros((z.shape[0],6))
            newz[:,[0,3,4,5]] = z[:,[0,5,6,7]]
            newz[:,1] = z[:,1] + z[:,3]
            newz[:,2] = z[:,2] + z[:,4]
            results.append(newz)
            del x, pos
            for v in newz:
                csvw_temp.writerow(v)
        if i % 100 == 0:
            print("{:d} processing time: ".format(i), time.time() - start0)

    print("{:d} processing time: ".format(i), time.time() - start)

    if results:
        return np.vstack(results)
    return None




def main(imagepath, outdir, resultname):
    '''
    run model on image stack

    Parameters:
    -----------
    imagepath : path to image stack to analyze

    outdir : directory to write result to

    resultname : name to give result csv file
    '''

    # Load RetinaNet model
    model_path = './trained_models/retinanet_models/model_final.pt'
    state_dict = torch.load(model_path)
    model_retinanet = retinanet_model.resnet50(1, False)
    model_retinanet.load_state_dict(state_dict)

    # Load ConvXYZ model
    model_path = './trained_models/convxyz_models/final_model.pt'
    state_dict = torch.load(model_path)
    model_convxyz = ConvXYZ(nrows = 32, ncols = 32, in_channels = 1, out_channels = 3, conv_dim = 64, drop_prob = 0.2, alpha = 0.1)
    model_convxyz.load_state_dict(state_dict)
    images = None

    # Load NphotonNet model
    model_path = './trained_models/convgamma_models/final_model.pt'
    state_dict = torch.load(model_path)
    model_gamma = ConvGamma()
    model_gamma.load_state_dict(state_dict)

    # Load images and Preprocess them
    images = skio.imread(imagepath)
    
    x = analyze(model_retinanet, model_convxyz, model_gamma, images, prob = 0.80,
    amplitude = 4095,# normalization amplitude
    min_side = 512,
    max_side = 1024,
    n_rows_convxyz = 32,
    n_cols_convxyz = 32,
    outputdirectory = outdir,
    resultname=resultname)

    if np.any(x):
        print("The result identifies {} events with the number of parameters {}".format(*x.shape) )
    else:
        print("No events are identified.")

if __name__ == '__main__':
    imagepath = './sample_data/sample.tiff'
    outdir = './sample_result'
    resultname = "result.csv"
    main(imagepath,outdir,resultname)
