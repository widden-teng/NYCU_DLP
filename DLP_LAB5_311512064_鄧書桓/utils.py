import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import random

def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def  plot_pred(validate_seq, validate_cond, modules, epoch, args):


    generation = [[] for i in range(5)]
    for num in range(5):
        h_t = [[] for i in range (args.n_past + args.n_future)]
        modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
        modules['posterior'].hidden = modules['posterior'].init_hidden()
        generation[num].append(validate_seq[0])
        save_image(validate_seq[0][0], "./img/" + str(num) + "/"+str(0)+".png")
        for l in range(0, args.n_past + 1):
            h_t[l] =  modules["encoder"](validate_seq[l])

        for i in range (1, args.n_past + args.n_future):
            if i < args.n_past:
                generation[num].append(validate_seq[i])
                save_image(validate_seq[i][0], "./img/"+str(num)+"/"+str(i+1)+".png")
                z_t, mu, logvar = modules['posterior'](h_t[i][0].detach())
                modules['frame_predictor'](torch.cat([validate_cond[i-1], h_t[i-1][0].detach(), z_t], 1)) 
            else:    
                z_t = torch.randn(args.batch_size, args.z_dim).cuda()
                g_t = modules['frame_predictor'](torch.cat([validate_cond[i-1] ,h_t[1-1][0].detach(), z_t], 1)).detach()
                x_bar_t = modules['decoder']([g_t, h_t[i-1][1]]).detach()
                h_t[i] = modules['encoder'](x_bar_t)
                generation[num].append(x_bar_t)
                save_image(x_bar_t[0], "./img/"+str(num)+"/"+str(i+1)+".png")


def pred(validate_seq, validate_cond, modules, args, device):
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    # if test: length = 30
    length = args.n_past + args.n_future
    x = []
    x.append(validate_seq[0])
    h_t = [[] for i in range (length)]
    for len in range(0, args.n_past + 1):
        h_t[len] =  modules["encoder"](validate_seq[len])
    for i in range (1, length):
        
        if args.last_frame_skip or i < args.n_past:	
            _, skip = h_t[i-1]
        
        if i < args.n_past:
            x.append(validate_seq[i])
            z_t, mu, logvar = modules['posterior'](h_t[i][0])
            modules['frame_predictor'](torch.cat([validate_cond[i-1], h_t[i-1][0].detach(), z_t], 1)) 
        else:    
            z_t = torch.randn(args.batch_size, args.z_dim).cuda()
            g_t = modules['frame_predictor'](torch.cat([validate_cond[i-1] ,h_t[i-1][0].detach(), z_t], 1)).detach()
            x_bar_t = modules['decoder']([g_t, skip]).detach()
            h_t[i] = modules['encoder'](x_bar_t)
            x.append(x_bar_t)
    return x        

def  plot_pred(test_seq, test_cond, modules, args):
    modules['posterior'].hidden = modules['posterior'].init_hidden()        
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    ground_truth = test_seq
    for idx in range(args.n_past + args.n_future):
        save_image(padded_image(ground_truth[idx][0], True, 'Ground \ntruth'), "./img/" + str(0) + "/"+str(idx)+".png")
        save_image(ground_truth[idx][0], "./img/GT/"+str(idx)+".png")
        
    save_image(padded_image(test_seq[0][0], True, 'Approx. \nposterior'), "./img/" + str(1) + "/"+str(0)+".png")
    x = test_seq[0]
    obs = True 
    for idx in range(1, args.n_past + args.n_future):
        if idx >= args.n_past:
            obs = False
              
        h = modules['encoder'](x)
        h_t = modules['encoder'](test_seq[idx])[0].detach()
        if args.last_frame_skip or idx < args.n_past:	
            _, skip = h
        else:
            _, _ = h  
        _, z_t, _= modules['posterior'](h_t) 
        if idx < args.n_past:
            modules['frame_predictor'](torch.cat([test_cond[idx-1], h[0].detach(), z_t], 1)) 
            x = test_seq[idx]
            save_image(padded_image(test_seq[idx][0], obs, 'Approx. \nposterior'), "./img/" + str(1) + "/"+str(idx)+".png")

        else:
            g_t = modules['frame_predictor'](torch.cat([test_cond[idx-1], h[0].detach(), z_t], 1)).detach()
            x_bar_t =  modules['decoder']([g_t, skip]).detach()
            save_image(padded_image(x_bar_t[0], obs, 'Approx. \nposterior'), "./img/" + str(1) + "/"+str(idx)+".png")
            x = x_bar_t
    psnr = np.zeros((args.batch_size, 5, args.n_future))    
    generation = [[] for i in range(5)]
    
    for num in range(3):
        test = []
        pred = []
        h_t = [[] for i in range (args.n_past + args.n_future)]
        modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
        modules['posterior'].hidden = modules['posterior'].init_hidden()
        obs = True
        generation[num].append(test_seq[0])

        # save_image(padded_image(test_seq[0][0], obs, f'Random\nSample {num + 1}'), "./img/" + str(num+3) + "/"+str(1)+".png")
        for l in range(0, args.n_past + 1):
            h_t[l] =  modules["encoder"](test_seq[l])

        for i in range (1, args.n_past + args.n_future):
            if i < args.n_past:
                obs = False
                generation[num].append(test_seq[i])
                # save_image(padded_image(test_seq[i][0], obs, f'Random\nSample {num + 1}'), "./img/"+str(num +3)+"/"+str(i+1)+".png")
                z_t, mu, logvar = modules['posterior'](h_t[i][0].detach())
                modules['frame_predictor'](torch.cat([test_cond[i-1], h_t[i-1][0].detach(), z_t], 1)) 
            else:    
                z_t = torch.randn(args.batch_size, args.z_dim).cuda()
                g_t = modules['frame_predictor'](torch.cat([test_cond[i-1] ,h_t[1-1][0].detach(), z_t], 1)).detach()
                x_bar_t = modules['decoder']([g_t, h_t[i-1][1]]).detach()
                h_t[i] = modules['encoder'](x_bar_t)
                generation[num].append(x_bar_t)
                pred.append(x_bar_t)
                test.append(test_seq[i])
                # save_image(padded_image(x_bar_t[0], obs, f'Random\nSample {num + 1}'), "./img/"+str(num+3)+"/"+str(i+1)+".png")
        _, _, psnr[:, num, :] = finn_eval_seq(test, pred)

    random_idx = random.sample([i for i in range(3)], 3)
    best_result_idx = np.argsort(np.mean(psnr[0], 1))[-1]
    for idx in range (args.n_past + args.n_future):
        if idx < args.n_past:
            obs = True
        else:
            obs = False 
        save_image(padded_image(generation[best_result_idx][idx][0], obs, 'Best PSNR'), "./img/"+str(2)+"/"+str(idx)+".png")

    for num in range(3):
        for idx in range (args.n_past + args.n_future):
            if idx < args.n_past:
                obs = True
            else:
                obs = False 
            if num == 0:    
                save_image(generation[random_idx[num]][idx][0], "./img/Pred/"+str(idx)+".png")    
            save_image(padded_image(generation[random_idx[num]][idx][0], obs, f'Random\nSample {num + 1}'), "./img/"+str(num+3)+"/"+str(idx)+".png")

def padded_image(image, obs, words):
    width = len(image[1])
    w = width + 27
    h = width + 2 
    new_image = torch.zeros(3, w, h)
    if obs:
        new_image[1] = 0.7
    elif not obs:
        new_image[0] = 0.7

    new_image[:, 2:width+2, 2:width+2] = image
    new_image = new_image.transpose(0, 1).transpose(1, 2).cpu().numpy() * 255
    new_image = Image.fromarray(np.uint8(new_image))

    draw = ImageDraw.Draw(new_image)
    draw.text((4, 64), words, (0, 0, 0))

    img_with_text = np.asarray(new_image) / 255.0
    img_with_text = torch.tensor(img_with_text).transpose(0, 2).transpose(1, 2)
        
    return img_with_text