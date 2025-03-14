import cupy as cp
import warnings

"""
[Parameters]
    psf : numpy.ndarray
    shape : tuple of int
[Returns]
    otf : numpy.ndarray
"""
def GenerateOTF(psf, shape):
    if cp.all(psf == 0):
        return cp.zeros_like(psf)
    
    inshape = psf.shape
    psf = ApplyZeroPadding(psf, shape)

    # Rotate OTF to make the 'center' of the PSF is [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = cp.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute OTF
    otf = cp.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error = machine epsilon
    n_ops = cp.sum(cp.array(psf.size) * cp.log2(cp.array(psf.shape)))
    otf = cp.real_if_close(otf, tol=n_ops)

    return otf

"""
[Parameters]
    image: real 2d numpy.ndarray
    shape: tuple of int
[Returns]
    pad_image : real 2d numpy.ndarray
"""
def ApplyZeroPadding(image, shape):
    shape = cp.asarray(shape, dtype=int)
    imshape = cp.asarray(image.shape, dtype=int)

    if cp.all(imshape == shape):
        return image
    
    if cp.any(shape <= 0):
        raise ValueError("ApplyZeroPadding: null or negative shape given")
    
    dshape = shape - imshape
    if cp.any(dshape < 0):
        raise ValueError("ApplyZeroPadding: target size is smaller than source")
    
    pad_image = cp.zeros(shape, dtype=image.dtype)
    idx, idy = cp.indices(imshape)

    pad_image[idx, idy] = image
    return pad_image

"""
[Parameters]
    NA : number
    wavelength : number
    dx : number
    dz : number
    shape : tuple of int
    n : number
[Returns]
    psf : numpy.ndarray
"""
def ComputeGaussianPSF(NA, wavelength, dx, dz, shape, n):
    Nx, Ny, Nz = shape
    z = cp.arange(-Nz, Nz + 1) * dz
    k = n / wavelength
    dk = cp.sqrt(2) * NA * k

    xi = cp.pi * dk ** 2 * z / 2 / k

    L = dx * Nx
    x = cp.arange(-L/2 + dx/2, L/2 + dx/2, dx)
    [xx, yy] = cp.meshgrid(x, x)
    rho = cp.sqrt(xx ** 2 + yy ** 2)

    psf = cp.zeros((Nx, Ny, z.shape[0]))
    
    for i in range(0, z.shape[0]):
        psf[:, :, i] = cp.pi * dk ** 2 / (1 + xi[i] ** 2) * cp.exp(-cp.pi ** 2 * dk ** 2 * rho ** 2 / (1 + xi[i] ** 2))
        psf[:, :, i] = psf[:, :, i] / cp.sum(psf[:, :, i])

    psf = psf / cp.sum(psf)
    return psf

"""
[Parameters]
    I : numpy.ndarray
    otf : numpy.ndarray
    maxIter : int
    reg : number
[Returns]
    J2 : numpy.ndarray
"""
def RL_TV(I, otf, maxIter = 20, reg = 0.01):
    # code-level check for possible RuntimeWarning (overflow, NaN etc)
    warnings.filterwarnings('error', category=RuntimeWarning)

    sizeI = cp.array(I.shape)
    J1 = cp.array(I)
    J2 = J1.copy()
    J3 = 0
    J4 = cp.zeros((cp.prod(sizeI).item(), 2))
    wI = cp.maximum(J1, 0)
    eps = cp.finfo(float).eps

    lda = 0
    for k in range(0, maxIter):
        print("[Inner Iteration] #", str(k+1), sep='')
        if k > 1:
            lda = cp.dot(J4[:, 0], J4[:, 1]) / cp.dot(J4[:, 1], J4[:, 1]) + eps
            # stability enforcement
            lda = cp.maximum(cp.minimum(lda, 1), 0)  
        Y = cp.maximum(J2 + lda * (J2 - J3), 0)
        
        # (3-b) make core for the LR estimation
        Reblurred = cp.fft.ifftn(otf * cp.fft.fftn(Y)).real
        Reblurred = cp.maximum(Reblurred, eps)
        ImRatio = wI / Reblurred + eps

        Ratio = cp.fft.ifftn(cp.conj(otf) * cp.fft.fftn(ImRatio)).real

        if reg != 0:
            TV_term = ComputeTV(J2, reg, eps)
            Ratio = Ratio / TV_term
        
        J3 = J2.copy()

        try:
            J2 = cp.maximum(Y * Ratio, 0)
        except RuntimeWarning:
            print(Y)
            print(Ratio)
            return
        
        J4 = cp.column_stack((J2.flatten() - Y.flatten(), J4[:, 0]))

    return J2

"""
[Parameters]
    I : numpy.ndarray
    reg : number
    eps : number
[Returns]
    TVterm : numpy.ndarray
"""
def ComputeTV(I, reg, eps):
    gx = cp.diff(I, axis=0)
    Oxp = cp.pad(gx, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    Oxn = cp.pad(gx, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    mx = (cp.sign(Oxp) + cp.sign(Oxn)) / 2 * cp.minimum(Oxp, Oxn)
    mx = cp.maximum(mx, eps)
    Dx = Oxp / cp.sqrt(Oxp ** 2 + mx ** 2)
    DDx = cp.diff(Dx, axis=0)
    DDx = cp.pad(DDx, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

    gy = cp.diff(I, axis=1)
    Oyp = cp.pad(gy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    Oyn = cp.pad(gy, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    my = (cp.sign(Oyp) + cp.sign(Oyn)) / 2 * cp.minimum(Oyp, Oyn)
    my = cp.maximum(my, eps)
    Dy = Oyp / cp.sqrt(Oyp ** 2 + mx ** 2)
    DDy = cp.diff(Dy, axis=1)
    DDy = cp.pad(DDy, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)

    TVterm = 1 - (DDx + DDy) * reg
    return cp.maximum(TVterm, eps)