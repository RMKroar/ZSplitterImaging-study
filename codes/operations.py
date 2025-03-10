import numpy as np

"""
[Parameters]
    psf : numpy.ndarray
    shape : tuple of int
[Returns]
    otf : numpy.ndarray
"""
def GenerateOTF(psf, shape):
    if np.all(psf == 0):
        return np.zeros_like(psf)
    
    inshape = psf.shape
    psf = ApplyZeroPadding(psf, shape)

    # Rotate OTF to make the 'center' of the PSF is [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error = machine epsilon
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

"""
[Parameters]
    image: real 2d numpy.ndarray
    shape: tuple of int
[Returns]
    pad_image : real 2d numpy.ndarray
"""
def ApplyZeroPadding(image, shape):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.all(imshape == shape):
        return image
    
    if np.any(shape <= 0):
        raise ValueError("ApplyZeroPadding: null or negative shape given")
    
    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ApplyZeroPadding: target size is smaller than source")
    
    pad_image = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(imshape)

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
    z = np.arange(-Nz, Nz + 1) * dz
    k = n / wavelength
    dk = np.sqrt(2) * NA * k

    xi = np.pi * dk ** 2 * z / 2 / k

    L = dx * Nx
    x = np.arange(-L/2 + dx/2, L/2 + dx/2, dx)
    [xx, yy] = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)

    psf = np.zeros((Nx, Ny, z.shape[0]))
    
    for i in range(0, z.shape[0]):
        psf[:, :, i] = np.pi * dk ** 2 / (1 + xi[i] ** 2) * np.exp(-np.pi ** 2 * dk ** 2 * rho ** 2 / (1 + xi[i] ** 2))
        psf[:, :, i] = psf[:, :, i] / np.sum(psf[:, :, i])

    psf = psf / np.sum(psf)
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
    sizeI = I.shape
    J1 = np.array(I)
    J2 = J1.copy()
    J3 = 0
    J4 = np.zeros((np.prod(sizeI), 2))
    wI = np.maximum(J1, 0)
    eps = np.finfo(float).eps

    lda = 0
    for k in range(0, maxIter):
        print("[Inner Iteration] #", str(k+1), sep='')
        if k > 1:
            lda = np.dot(J4[:, 0], J4[:, 1]) / np.dot(J4[:, 1], J4[:, 1]) + eps
            # stability enforcement
            lda = np.maximum(np.minimum(lda, 1), 0)     
        Y = np.maximum(J2 + lda * (J2 - J3), 0)
        
        # (3-b) make core for the LR estimation
        Reblurred = np.fft.ifftn(otf * np.fft.fftn(Y)).real
        Reblurred = np.maximum(Reblurred, eps)
        ImRatio = wI / Reblurred + eps

        Ratio = (np.fft.ifftn(np.conj(otf)) * np.fft.fftn(ImRatio)).real
        if not reg == 0:
            TV_term = ComputeTV(J2, reg, eps)
            Ratio = Ratio / TV_term
        
        J3 = J2.copy()
        J2 = np.maximum(Y * Ratio, 0)
        J4 = np.column_stack((J2.ravel() - Y.ravel(), J4[:, 0]))

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
    gx = np.diff(I, axis=0)
    Oxp = np.pad(gx, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    Oxn = np.pad(gx, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    mx = (np.sign(Oxp) + np.sign(Oxn)) / 2 * np.minimum(Oxp, Oxn)
    mx = np.maximum(mx, eps)
    Dx = Oxp / np.sqrt(Oxp ** 2 + mx ** 2)
    DDx = np.diff(Dx, axis=0)
    DDx = np.pad(DDx, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

    gy = np.diff(I, axis=1)
    Oyp = np.pad(gy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    Oyn = np.pad(gy, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    my = (np.sign(Oyp) + np.sign(Oyn)) / 2 * np.minimum(Oyp, Oyn)
    my = np.maximum(my, eps)
    Dy = Oyp / np.sqrt(Oyp ** 2 + mx ** 2)
    DDy = np.diff(Dy, axis=1)
    DDy = np.pad(DDy, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)

    TVterm = 1 - (DDx + DDy) * reg
    return np.maximum(TVterm, eps)