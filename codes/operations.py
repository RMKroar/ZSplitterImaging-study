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