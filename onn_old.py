import torch

import numpy as np





def detector_region(x):

    return torch.cat((

        x[:, 46 : 66, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),

        x[:, 46 : 66, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),

        x[:, 46 : 66, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),

        x[:, 85 : 105, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),

        x[:, 85 : 105, 78 : 98].mean(dim=(1, 2)).unsqueeze(-1),

        x[:, 85 : 105, 109 : 129].mean(dim=(1, 2)).unsqueeze(-1),

        x[:, 85 : 105, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),

        x[:, 125 : 145, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),

        x[:, 125 : 145, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),

        x[:, 125 : 145, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)





class DiffractiveLayer(torch.nn.Module):



    def __init__(self):

        super(DiffractiveLayer, self).__init__()

        self.size = 200                         # 200 * 200 neurons in one layer

        self.distance = 0.03                    # distance bewteen two layers (3cm)

        self.ll = 0.08                          # layer length (8cm)

        self.wl = 3e8 / 0.4e12                  # wave length

        self.fi = 1 / self.ll                   # frequency interval

        self.wn = 2 * 3.1415926 / self.wl       # wave number

        # self.phi (200, 200)

        self.phi = np.fromfunction(

            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),

            shape=(self.size, self.size), dtype=np.complex64)

        # h (200, 200)

        h = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * self.phi))

        # self.h (200, 200, 2)

        self.h = torch.nn.Parameter(torch.stack((torch.from_numpy(h.real), torch.from_numpy(h.imag)), dim=-1), requires_grad=False)



    def forward(self, waves):

        # waves (batch, 200, 200, 2)

        #temp = torch.fft.fftn(waves, signal_ndim=2)
        temp = torch.fft.fftn(waves)

        k_pace_real = self.h[..., 0] * temp[..., 0] - self.h[..., 1] * temp[..., 1]

        k_space_imag = self.h[..., 0] * temp[..., 1] + self.h[..., 1] * temp[..., 0]

        k_space = torch.stack((k_pace_real, k_space_imag), dim=-1)

        # angular_spectrum (batch, 200, 200, 2)

        #angular_spectrum = (torch.fft.ifftn(k_space, signal_ndim=2))
        angular_spectrum = (torch.fft.ifftn(k_space))

        return angular_spectrum



"""

def _propogation(u0, d=delta, N = size, dL = dL, lmb = c/Hz,theta=0.0):

    #Parameter 

    df = 1.0/dL

    k = np.pi*2.0/lmb

    D= dL*dL/(N*lmb)

  

    #phase

    def phase(i,j):

        i -= N//2

        j -= N//2

        return ((i*df)*(i*df)+(j*df)*(j*df))

    ph  = np.fromfunction(phase,shape=(N,N),dtype=np.float32)

    #H

    H = np.exp(1.0j*k*d)*np.exp(-1.0j*lmb*np.pi*d*ph) 

    #Result

    return tf.ifft2d(np.fft.fftshift(H)*tf.fft2d(u0)*dL*dL/(N*N))*N*N*df*df

  

def propogation(u0,d,function=_propogation):

    return tf.map_fn(function,u0)



"""

class Net(torch.nn.Module):

    """

    phase only modulation

    """

    def __init__(self, num_layers=5):



        super(Net, self).__init__()

        # self.phase (200, 200)

        # np.random.random(size=(200, 200)).astype('float32')

        self.phase = [torch.nn.Parameter(torch.from_numpy( 2 * np.pi * torch.nn.init.xavier_uniform_(torch.empty(200,200)).numpy() ), requires_grad=True) for _ in range(num_layers)]

        for i in range(num_layers):

            self.register_parameter("phase" + "_" + str(i), self.phase[i])

        self.diffractive_layers = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])

        self.last_diffractive_layer = DiffractiveLayer()

        #self.softmax = torch.nn.Softmax(dim=-1)

        self.softmax = torch.nn.LogSoftmax(dim=-1)



    def forward(self, x):

        # x (batch, 200, 200, 2)

        for index, layer in enumerate(self.diffractive_layers):

            temp = layer(x)

            exp_j_phase = torch.stack((torch.cos(self.phase[index]), torch.sin(self.phase[index])), dim=-1)

            x_real = temp[..., 0] * exp_j_phase[..., 0] - temp[..., 1] * exp_j_phase[..., 1]

            x_imag = temp[..., 0] * exp_j_phase[..., 1] + temp[..., 1] * exp_j_phase[..., 0]

            x = torch.stack((x_real, x_imag), dim=-1)

        x = self.last_diffractive_layer(x)
        print(x.shape, "shape of x")
        print(x)

        # x_abs (batch, 200, 200)

        x_abs = torch.sqrt(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1])

        output = self.softmax(detector_region(x_abs))

        #output = (detector_region(x_abs))
        print(output.shape, "shape of output")
        print(output)
        


        return output





if __name__ == '__main__':

    print(Net())

