import torch
import numpy as np
from numpy.linalg import multi_dot


def detector_region(x):
    return torch.cat((
        x[:, 46: 66, 46: 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46: 66, 93: 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46: 66, 140: 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85: 105, 46: 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85: 105, 78: 98].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85: 105, 109: 129].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85: 105, 140: 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125: 145, 46: 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125: 145, 93: 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125: 145, 140: 160].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)


class DiffractiveLayer(torch.nn.Module):

    def __init__(self):
        super(DiffractiveLayer, self).__init__()
        self.size = 200  # 200 * 200 neurons in one layer
        self.distance = 0.03  # distance between two layers (3cm)
        self.ll = 0.08  # layer length (8cm)
        self.wl = 3e8 / 0.4e12  # wave length
        self.fi = 1 / self.ll  # frequency interval
        self.wn = 2 * 3.1415926 / self.wl  # wave number
        # self.phi (200, 200)
        self.phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size), dtype=np.complex64)
        # h (200, 200)
        # forward propagation
        hf = np.fft.fftshift(
            np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * self.phi))
        # reverse propagation
        hr = np.fft.fftshift(
            np.exp(1.0j * self.wn * self.distance) * np.exp(+1.0j * self.wl * np.pi * self.distance * self.phi))

        # self.h (200, 200, 2)
        self.hf = torch.nn.Parameter(torch.stack((torch.from_numpy(hf.real), torch.from_numpy(hf.imag)), dim=-1),
                                     requires_grad=False)
        self.hr = torch.nn.Parameter(torch.stack((torch.from_numpy(hr.real), torch.from_numpy(hr.imag)), dim=-1),
                                     requires_grad=False)

    def forward(self, wavesf, wavesr):
        # waves (batch, 200, 200, 2)
        tempf = torch.fft(wavesf, signal_ndim=2)
        tempr = torch.fft(wavesr, signal_ndim=2)
        
        #print(tempf.shape, "shape of the tempf in first forward")
       
   
        
        kf_space_real = self.hf[..., 0] * tempf[..., 0] - self.hf[..., 1] * tempf[..., 1]
        kf_space_imag = self.hf[..., 0] * tempf[..., 1] + self.hf[..., 1] * tempf[..., 0]
        kf_space = torch.stack((kf_space_real, kf_space_imag), dim=-1)

        kr_space_real = self.hr[..., 0] * tempr[..., 0] - self.hr[..., 1] * tempr[..., 1]
        kr_space_imag = self.hr[..., 0] * tempr[..., 1] + self.hr[..., 1] * tempr[..., 0]
        kr_space = torch.stack((kr_space_real, kr_space_imag), dim=-1)

        # k_space = torch.stack((kf_space, kr_space), dim=-1)
        # angular_spectrum (batch, 200, 200, 2)
        angular_spectrumf = torch.ifft(kf_space, signal_ndim=2)
        angular_spectrumr = torch.ifft(kr_space, signal_ndim=2)
        return angular_spectrumf, angular_spectrumr


class Net(torch.nn.Module):
    """
    phase only modulation
    """

    def __init__(self, num_layers=5):

        super(Net, self).__init__()
        # self.phase (200, 200)
        self.na = 1
        self.ns = 1.5
        self.size = 200
        self.phase = [torch.nn.Parameter(torch.from_numpy( 2 * np.pi * torch.nn.init.xavier_uniform_(torch.empty(200,200)).numpy() ), requires_grad=True) for _ in range(num_layers)]

       # self.phase = [torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=(200, 200)).astype('float32'))) for _in range(num_layers)]

        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), self.phase[i])

        self.diffractive_layers = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])

        self.last_diffractive_layer = DiffractiveLayer()

        # self.sofmax = torch.nn.Softmax(dim=-1)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def model(self, xf, xr):
        # x (batch, 200, 200, 2)
        self.na = 1
        self.ns = 1.5

        for index, layer in enumerate(self.diffractive_layers):
            tempf, tempr = layer(xf, xr)
            #print('*'*100)
            #print(tempf.shape,"tempf in the model")
            #print('*' *100)
            #print(tempf.shape, tempf.item())

            #tempf = tempf.type(torch.complex64)
            #tempr = tempr.type(torch.complex64)


            t = (self.ns - self.na) / (self.ns + self.na)
            r = (2 * self.na) / (self.ns + self.na)

            exp_jp_phase = torch.stack((torch.cos(self.phase[index]), torch.sin(self.phase[index])), dim=-1)
            exp_jn_phase = torch.stack((torch.cos(self.phase[index]), -torch.sin(self.phase[index])), dim=-1)
            
            exp_jp_phase = exp_jp_phase.detach().numpy()
            exp_jn_phase = exp_jn_phase.detach().numpy()
            
            #print(exp_jn_phase.shape, "shape of exp_phase")

            for i in range(0, self.size-1):
                for j in range (0, self.size-1):
                    x = np.divide(1, t)
                    y = np.divide(r, t)
                    w = np.divide(r, t)
                    z = np.divide(1, t)
                
                    #print(w.shape, "shape of w")
            
                    matrix_r = np.array ([[x,y],[w,z]]) # Transfer matrix associated with starting interface
            
            #atrix_r = np.array([[x, y], [w, z]])  

                    e = np.divide(1, t)
                    f = np.divide(-r, t)
                    g = np.divide(-r, t)
                    h = np.divide(1, t)
                    #print(e.shape, "shape of e")
                    #matrix_t = torch.tensor ([[e,f],[g,h]])
                
                    matrix_t = np.array([[e, f], [g, h]])  # Transfer matrix associated with end interface
            
                    t11 = exp_jn_phase[i,j]
                    t12 = 0 #torch.zeros((self.size, self.size))
                    t21 = 0 #torch.zeros((self.size, self.size))
                    t22 = exp_jp_phase[i,j]
                    matrix_m = np.array([[t11, t12], [t21, t22]])  # Transfer matrix associated to the layer between interfaces
                    #matrix_m = torch.tensor ([[t11,t12],[t21,t22]])
                    #t_matrix = matrix_r@matrix_m@matrix_t
                    t_matrix = multi_dot([matrix_r, matrix_m, matrix_t])
                    
                    return t_matrix
            
            #atrix_m = np.array([[t11, t12], [t21, t22]])  # Transfer matrix associated to the layer between interfaces

            t_matrix = torch.tensor(t_matrix)
            print(t_matrix.shape, "shape of t_matrix")

            a = t_matrix.item(0, 0)
            #a = a.type(torch.complex64)
            print(a.shape, "shape of a")

            b = t_matrix.item(0, 1)
            #b = b.type(torch.complex64)
            #print(b.shape, "shape of b")

            c = t_matrix.item(1, 0)
            #c = c.type(torch.complex64)
            #print(c.shape, "shape of c")

            d = t_matrix.item(1, 1)
            #d = d.type(torch.complex64)
           # print(d.shape, "shape of d")

            # t = (1 - self.gamma ** 2) * exp_j_phase
            # r = self.gamma * (1-2*exp_j_phase)
            # a = np.linalg.inv(numpy.transpose(t))
            # b = np.divide(r, t)
            # c = np.divide(numpy.transpose(r), numpy.transpose(t))
            # d = np.linalg.inv(t)

            xp_real = (a[..., 0] * tempf[..., 0] - a[..., 1]* tempf[..., 1]) + (b[..., 0] * tempr[..., 0] - b[..., 1] * tempr[..., 1])
            #print(xp_real.shape, "shape of xp_real")
            
            xp_imag = (a[..., 0]* tempf[..., 1] + a[..., 1] * tempf[..., 0]) + (b[..., 0] * tempr[..., 1] + b[..., 1] * tempr[..., 0])

            xn_real = (c[..., 0] * tempf[..., 0] - c[..., 1] * tempf[..., 1]) + (d[..., 0] * tempr[..., 0] - d[..., 1]* tempr[..., 1])
            xn_imag = (c[..., 0] * tempf[..., 1] + c[..., 1] * tempf[..., 0]) + (d[..., 0] * tempr[..., 1] + d[..., 1] * tempr[..., 0])

            xp = torch.stack((xp_real, xp_imag), dim=-1)
            
            #print(xp.shape, "shape of xp")
            xn = torch.stack((xn_real, xn_imag), dim=-1)

            xf,xr = self.last_diffractive_layer(xp,xn)
            #xr = self.last_diffractive_layer(xn)
            
            #print(xf.shape, "shape of xf")

            return xf, xr
        # x_abs (batch, 200, 200)

    def forward(self, xf):

        a, b = self.model(torch.ones((self.size, self.size,2)), torch.zeros((self.size, self.size,2)))
        c, d = self.model(torch.zeros((self.size, self.size,2)), torch.ones((self.size, self.size,2)))

        print(a.shape, "shape of a in forward")
        print(b.shape, "shape of b in forward")
        print(c.shape, "shape of c in forward")
        print(d.shape, "shape of d in forward")
     
    
        #output=a*input-bc/d*input
        
        b_c=torch.matmul(b, c)
        print(b_c.shape, "shape of b_c in forward")
        
        di_v=torch.div(b_c,d)
        #print(di_v.shape, "shape of di_v in forward")
        
        fin=a-di_v
        #print(fin.shape, "shape of fin in forward")
        
        yf=fin@xf

        #yf = torch.matmul(a, xf) - torch.matmul(torch.div(torch.matmul(b, c), d), xf)
        #print(yf.shape, "shape of yf")

        yf_abs = torch.sqrt(yf[..., 0] * yf[..., 0] + yf[..., 1] * yf[..., 1])

        # yr_abs = torch.sqrt(yr[..., 0] * yr[..., 0] + yr[..., 1] * yr[..., 1])

        output = self.softmax(detector_region(yf_abs))

        # outputr = self.sofmax(detector_region(yr_abs))
        return output


if __name__ == '__main__':
    print(Net())
