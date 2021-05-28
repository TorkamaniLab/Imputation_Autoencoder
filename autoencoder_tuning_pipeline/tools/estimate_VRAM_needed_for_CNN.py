import torch.nn as nn
import torch
from torch.autograd import Variable
import sys
#from torchsummary import summary
import subprocess
import torch.cuda as cutorch
import os


if len(sys.argv) <= 1:

    print("Usage: CUDA_VISIBLE_DEVICES=<GPUID> python3 estimate_VRAM_needed_for_autoencoder.py <N_VARIANTS>")
    print("    <GPUID>: GPU ID or GPU index (i.e. 0)")
    print("    <N_VARIANTS>: number of genetic variants in the VMV/VCF file")
    print("Example: CUDA_VISIBLE_DEVICES=0 python3 estimate_VRAM_needed_for_autoencoder.py 6000")

    sys.exit()


#SPARSE DENOISING AUTOENCODER, DIMENSIONALITY REDUCTION WITH SPARSITY LOSS (KL-DIVERGENCE)
def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).to('cuda:0')
    return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

def l1_loss(model):
    l1_regularization = 0.
    l1_loss = nn.L1Loss()
    for name, param in model.named_parameters():
        if 'weight' in name:
            #faster
            l1_regularization = l1_regularization + l1_loss(param, target=torch.zeros_like(param))
            #torch.norm(x, ord) = sum(abs(x)**ord)**(1./ord)
            #l1_regularization = l1_regularization + torch.norm(param, 1)
    return l1_regularization

def sparse_loss(rho, data, model_children):
    #apply sparsity on encoder and decoder
    #values = data
    #loss = 0
    #for i in range(len(model_children)):
    #    values = model_children[i](values)
    #    loss += kl_divergence(rho, values)
    #aply sparsity on encoder only
    encoded_values = model_children[0](data.unsqueeze(1)).unsqueeze(0)
    loss = kl_divergence(rho, encoded_values)
    return loss


#sparse autoencoder, keep full hidden layer size equal output_dim, then apply sparsity in the loss later
class Autoencoder(nn.Module):
    def __init__(self,input_dim, output_dim, n_layers=4, size_ratio=0.5, activation='relu', kernel_size=5, pool_size=2, drop_rate=0.25, max_nkernel_limit=256):
        super(Autoencoder, self).__init__()

        def get_activation(activation):

            if(activation=='relu'):
                return nn.ReLU(True)
            elif(activation=='tanh'):
                return nn.Tanh()
            elif(activation=='sigmoid'):
                return nn.Sigmoid()
            elif(activation=='leakyrelu'):
                return torch.nn.LeakyReLU()

        #nn.Linear Parameters
        #input_dim – size of each input sample
        #output_dim – size of each output sample
        #n_layers - total number of layers
        #output layer channels to restore initial shape
        n_channels=1
        #padding
        P=2
        #maximum number of kernels considering 32GB VRAM and runtime limits
        #max_nkernel_limit = 256

        out_size_list = [min(int(output_dim*size_ratio),max_nkernel_limit)]
        in_size_list = []
        for i in range(int(n_layers/2)):
            out_size_list += [int(out_size_list[i]*size_ratio)]
            in_size_list += [out_size_list[i+1]]

        i=i+1
        out_size_list += [int(out_size_list[i]*size_ratio)]
        in_size_list += [out_size_list[i+1]]
        in_size_list += [n_channels]
        out_size_list[0] = n_channels

        out_size_list.reverse()
        in_size_list.reverse()

        encoder_layers = []
        for i in range(int(n_layers/2)):
            encoder_layers += [nn.Conv1d(in_channels=in_size_list[i], out_channels=out_size_list[i], kernel_size=kernel_size, padding=P)]
            encoder_layers += [get_activation(activation)]
            encoder_layers += [nn.MaxPool1d(kernel_size=pool_size, ceil_mode=True)]
            encoder_layers += [nn.Dropout(drop_rate)]
            print(in_size_list[i],out_size_list[i])

        i=i+1
        encoder_layers += [nn.Conv1d(in_channels=in_size_list[i], out_channels=out_size_list[i], kernel_size=kernel_size, padding=P)]
        encoder_layers += [get_activation(activation)]

        decoder_layers = []
        out_size_list.reverse()
        in_size_list.reverse()
        out_size_list[0]=in_size_list[0]

        for i in range(int(n_layers/2)):
            decoder_layers += [nn.Conv1d(in_channels=out_size_list[i], out_channels=in_size_list[i], kernel_size=kernel_size, padding=P)]
            decoder_layers += [get_activation(activation)]
            if(i == (int(n_layers/2)-1)):
                decoder_layers += [nn.Upsample(output_dim)]
            else:
                decoder_layers += [nn.Upsample(scale_factor=pool_size)]
            decoder_layers += [nn.Dropout(drop_rate)]

        i=i+1
        decoder_layers += [nn.Conv1d(in_channels=out_size_list[i], out_channels=n_channels, kernel_size=kernel_size, padding=P)]
        decoder_layers += [get_activation('sigmoid')]

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


ni=int(sys.argv[1])*2
batch_size=64
nl=4
sr=0.5

print("SIMULATION PARAMETERS (WORST CASE SCENARIO FROM GRID SEARCH)")
print("number of variants:", sys.argv[1])
print("number of input/output nodes:", ni)
print("batch size:", batch_size)
print("number of layers:", nl)
print("size ratio:", sr)
print("Optimizer: Adam")

autoencoder = Autoencoder(input_dim=ni, output_dim=ni, n_layers=nl, size_ratio=sr, activation='tanh').cuda()

criterion = nn.BCELoss()
optimizer=torch.optim.Adam(autoencoder.parameters(), lr=0.0001, weight_decay=0.001)
#optimizer=torch.optim.RMSprop(autoencoder.parameters(), lr=0.0001, weight_decay=0.001)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)
model_children = list(autoencoder.children()) 

for i in range(3):
    masked_data  = torch.FloatTensor(batch_size,ni)
    true_data = torch.FloatTensor(batch_size,ni)

    masked_data = torch.where(masked_data > 0, torch.ones(batch_size,ni), torch.zeros(batch_size,ni))
    true_data = torch.where(true_data > 0, torch.ones(batch_size,ni), torch.zeros(batch_size,ni))

    #masked_data = Variable(masked_data).cuda()
    masked_data = torch.unsqueeze(Variable(masked_data).cuda(),1)
    true_data = Variable(true_data).cuda()

    reconstructed = autoencoder(masked_data)
    reconstructed = torch.squeeze(reconstructed)
    loss = criterion(reconstructed, true_data)
    l1_sparsity = l1_loss(autoencoder)
    loss = loss + l1_sparsity
    kl_sparsity = sparse_loss(0.05, true_data, model_children)
    loss = loss + kl_sparsity
    loss.backward()
    optimizer.step()
    my_lr_scheduler.step()


#this package only tells VRAM occupied by the pytorch variables, it doesn't include the extra driver/buffer VRAM that nvidia-smi allocates
#print(cutorch.max_memory_allocated(0))
#print(cutorch.max_memory_cached(0))
#1MiB=1048576bytes
print("GPU RAM for pytorch session only:", str(cutorch.max_memory_reserved(0)/1048576)+"MiB")

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi' #, '--query-gpu=memory.used',
            #'--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    PID=os.getpid()
    #print(result)
    gpu_memory = [x for x in result.strip().split('\n')]
    #print("looking for PID", PID)
    for line in gpu_memory:
        if str(PID) in line:
            print("GPU RAM including extra driver buffer from nvidia-smi:", line.split(' ')[-2])

get_gpu_memory_map()

