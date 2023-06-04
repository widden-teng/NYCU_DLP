import re
import matplotlib.pyplot as plt
import numpy as np 

niter = 300
kl_anneal_cycle = 3
kl_anneal_cyclical = False
class kl_annealing():
    def __init__(self, ratio):
        super().__init__()
        self.iter = 0
        self.beta = np.ones(niter) * 1.0
        if kl_anneal_cyclical:
            period = niter/ kl_anneal_cycle
            step = (1.0 - 0.0) / (period * ratio) 
            for c in range(kl_anneal_cycle):
                v, i = 0.0, 0.0 
                while v <= 1.0 and (int(self.iter +c*period) < niter):
                    self.beta[int(i + c * period)] = v
                    v += step
                    i  += 1
        else:
            period = niter / 1
            step = (1.0 - 0) / (period * ratio)
            v, i, c = 0.0, 0.0, 0.0
            while v <= 1.0 and (int(i + c * period) < niter):
                self.beta[int(i + c * period)] = v
                v += step
                i += 1            
    
    def update(self):
        self.iter += 1
    
    def get_beta(self):
        return self.beta[self.iter]
    
def plot_log(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
        epoch_list = [] 
        loss_list = []
        mse_list = []
        kld_list = []
        psnr_list = []
        teacher_list = [] 
        kl_beta_list = []
        for line in lines:
            if "validate psnr" in line:
                psnr = float(re.findall(r"\d+\.\d+", line)[0])
                psnr_list.append(psnr)
            elif "loss:" in line:
                logs = re.findall(r"\d+\.\d+", line)
                epoch = int(re.findall(r"\d+", line)[0])
                loss, mse_loss, kld_loss, teacher = logs
                epoch_list.append(float(epoch))
                loss_list.append(float(loss))
                mse_list.append(float(mse_loss))
                kld_list.append(float(kld_loss))
                teacher_list.append(float(teacher))
                # kl_beta_list.append(float(KLratio))
        kl_anneal = kl_annealing(ratio=0.5)  
        for _ in range (len(epoch_list)):
            kl_beta_list.append(kl_anneal.get_beta())
            kl_anneal.update()
            

    
        psnr_epoch_list = [i for i in range(len(epoch_list))  if i % 5 == 0]
    
        fig, ax1 = plt.subplots()
        plt.title('Training loss/ratio curve')
        plt.xlabel(f'{str(len(epoch_list))} iteration(s)')
        ax1.plot(epoch_list, loss_list, 'y-.', label='Loss')
        ax1.plot(epoch_list, kld_list, 'b-', label='KLD')
        ax1.plot(psnr_epoch_list, psnr_list, 'g.', label='PSNR')
        ax1.plot(epoch_list, mse_list,'m--', label='MSE')
        ax1.legend()
        ax1.set_ylabel('Loss/PSNR', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(epoch_list, teacher_list, 'k--', label='Teacher Ratio')
        ax2.plot(epoch_list, kl_beta_list, 'r-.', label='KL Weight')
        ax2.set_ylabel('Scores/ Weight', color='r')
        ax2.tick_params('y', colors='r')
        ax2.legend()
        fig.tight_layout()
        # plt.show()
        # ax1.set_ylim([0.0, 30.0])
        plt.savefig('plot_False.png')


plot_log("./logs/fp/lr0.002_tr_step100/train_record.txt")

    