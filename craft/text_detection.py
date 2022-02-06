import time
import os

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import cv2
import numpy as np
from craft import craft_utils, imgproc, file_utils

from craft.craft import CRAFT

from collections import OrderedDict

import matplotlib.pyplot as plt

from django.conf import settings


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(args,net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)

    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0))

    return boxes, polys, ret_score_text


class Args():
    def __init__(self, cuda=False, trained_model='craft/weights/craft_mlt_25k.pth', text_threshold=0.7, low_text=0.4,
                 link_threshold=0.4, canvas_size=1280, mag_ratio=1.5, poly=False, show_time=False, test_folder='/data/',
                 refine=False, refiner_model='craft/weights/craft_refiner_CTW1500.pth'):
        self.cuda = cuda
        self.trained_model = trained_model
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        self.show_time = show_time
        self.test_folder = test_folder
        self.refine = refine
        self.refiner_model = refiner_model


def img_show(img, size=(15, 15)):
    plt.rcParams["figure.figsize"] = size
    imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def get_text_detection(imgs, save_img_file=True):
    args = Args(test_folder='./', text_threshold=0.8, link_threshold=0.4, canvas_size=1800,
                refine=False, poly=False)

    result_folder = settings.SAVE_IMG_PATH
    # load net
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from craft.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()
    # load data
    for k, image_path in enumerate(imgs):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(imgs), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                                             args.cuda, args.poly, refine_net, args=args)
        if save_img_file:
            copy_img = image.copy()
            for pts in bboxes:
                pts = pts.reshape(-1, 2).astype(np.int32)
                copy_img = cv2.polylines(copy_img, [pts], True, (255, 0, 0), 3)

            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = result_folder + "res_" + filename + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)
            print(filename)

            img = file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))

    return bboxes


