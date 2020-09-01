import numpy as np
from scipy import special
from scipy import signal
import matplotlib.pyplot as plt
from skimage import draw
from matplotlib.widgets import Slider, Button, RadioButtons
from PIL import Image
from skimage.util import img_as_ubyte, img_as_float, img_as_int
import copy
def somb(x):
    z=np.zeros(np.shape(x));
    x = abs(x)
    idx = np.argwhere(x)
    z[idx] = 2*special.jv(1,np.pi*x[idx])/(np.pi*x[idx]);
    return z
condensers = {
    "Ph1": (0.45, 3.75, 24),
    "Ph2": (0.8, 5.0, 24),
    "Ph3": (1.0, 9.5, 24),
    "Ph4": (1.5, 14.0, 24),
    "PhF": (1.5, 19.0, 25)
} #W, R, Diameter

def get_kernel(scale,radius,F,condenser):
    W, R, _ = condensers[condenser]
    W = W*scale
    R = R*scale
    F = F*scale
    Lambda = 0.5
    xx,yy = np.meshgrid(np.linspace(-radius,radius,2*radius+1), np.linspace(-radius,radius,2*radius+1), sparse=True)
    scale2 = 10
    xx = xx/scale2
    yy = yy/scale2
    rr = np.sqrt(xx**2 + yy**2)
    rr_dl = rr*(2*np.pi)*(1/F)*(1/Lambda)
    kernel1 = np.pi * R**2 * somb(2*R*rr_dl)
    kernel2 = np.pi * (R - W)**2 * somb(2*(R - W)*rr_dl)
    kernel = np.nan_to_num(kernel1) - np.nan_to_num(kernel2)
    kernel = -kernel/np.linalg.norm(kernel)
    kernel[radius,radius] = kernel[radius,radius] + 1
    diraced = copy.deepcopy(kernel)
    diraced[np.where(diraced == 0)] = 1
    kernel = -kernel - diraced
    return kernel

scale = 9000
mult=2.4
radius = 5

def get_image(scale=scale,mult=mult,radius=radius, condenser="Ph2",resolution=100,F=60):
    resolution = int(resolution)
    radius = radius*scale/5000 * resolution/100
    kernel = get_kernel(scale,10,F,condenser)
    img = np.zeros((resolution, resolution, 1), dtype=np.double) + 0
    spacing = mult*radius
    for x in range(int(radius+radius/2), int(resolution+radius), int(spacing)):
        for y in range(int(radius+radius/2),int(resolution+radius), int(spacing)):
            rr, cc = draw.circle(x+x*np.random.rand(), y+y*np.random.rand(), radius, shape=img.shape)
            img[rr,cc] = 1
    img = img.reshape(resolution,resolution)
    convolved = signal.convolve2d(img, kernel,mode="same",boundary="fill")
    print("max {}, min {}".format(np.max(kernel), np.min(kernel)))
    return (kernel)
fig, ax = plt.subplots()
l= plt.imshow(get_image(),cmap="Greys")
plt.axis("off")
plt.title("Phase contrast microscope interactive model")
ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'

axmult = plt.axes([0.13,0.0,0.65,0.03], facecolor=axcolor)
axscale = plt.axes([0.13, 0.03, 0.65, 0.03], facecolor=axcolor)
axradius = plt.axes([0.13, 0.06, 0.65, 0.03], facecolor=axcolor)
axresolution = plt.axes([0.13, 0.09, 0.65, 0.03], facecolor=axcolor)
axF= plt.axes([0.13,0.12,0.65,0.03], facecolor=axcolor)
rax = plt.axes([0.025, 0.5, 0.12, 0.15], facecolor=axcolor)

smult = Slider(axmult, 'mult', 1, 10, valinit=mult)
sscale = Slider(axscale, 'scale', 1, 10000, valinit=scale)
sradius = Slider(axradius, 'radius', 1, 50, valinit=radius)
sresolution = Slider(axresolution, "image res", 20,500, valinit=100)
rcondense = RadioButtons(rax, ('Ph1', 'Ph2', 'Ph3', "Ph4", "PhF"), active=1)
sF = Slider(axF, 'Focal length', 20, 100, valinit=60)

def update(var):
    mult = smult.val
    scale = sscale.val
    radius = sradius.val
    resolution = sresolution.val
    F = sF.val
    l.set_data(get_image(scale=sscale.val,mult=smult.val,radius=sradius.val,resolution=sresolution.val,condenser=rcondense.value_selected,F=sF.val))
    def PhFunc(label):
        l.set_data(get_image(scale=sscale.val,mult=smult.val,radius=sradius.val,resolution=sresolution.val,condenser=label,F=sF.val))
        fig.canvas.draw_idle()
    rcondense.on_clicked(PhFunc)
    #fig.canvas.draw_idle()


smult.on_changed(update)
sscale.on_changed(update)
sradius.on_changed(update)
sresolution.on_changed(update)
sF.on_changed(update)
rcondense.on_clicked(update)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    smult.reset()
    sscale.reset()
    sradius.reset()
button.on_clicked(reset)

plt.show()