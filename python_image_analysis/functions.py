def random_graphene_with_vacancies(atoms, p=.05, extent):
    
    graphene.rotate('z', 25, rotate_cell=True)

    graphene = cut_rectangle(graphene, extent = (110, 110), origin = (0, 0))
    
    defect_graphene = atoms.copy()
    
    vacancies = np.where(np.random.rand(len(defect_graphene)) < p)
    
    vacancy_positions = defect_graphene.positions[vacancies]
    
    del defect_graphene[vacancies]
    
    return defect_graphene, vacancy_positions[:, :2]


def scherzer_ctf(Cs, energy):
    defocus = ab.transfer.scherzer_defocus(energy=wave.energy, Cs=Cs)
    
    wavelength = ab.utils.energy2wavelength(ctf.energy)
    
    semiangle_cutoff = 1000 * wavelength / ab.transfer.point_resolution(ctf.Cs, ctf.energy)
    
    return ab.CTF(Cs=Cs, defocus=defocus, energy=wave.energy, semiangle_cutoff=semiangle_cutoff)


def simulate_noisy_hrtem_image(atoms, dose, Cs, energy=80e3, sampling=.2):
    exit_wave = ab.PlaneWave(energy=energy).multislice(atoms, pbar=False)
    
    ctf = scherzer_ctf(Cs, energy)
    
    image = exit_wave.apply_ctf(ctf).intensity()
    
    noisy_image = poisson_noise(image, dose=dose)
    
    return noisy_image.array


def pixel_distance_to_first_spot(n, pixel_size, lattice_constant):
    return 1 / lattice_constant * 2 / np.sqrt(3) * n * pixel_size * 2 * np.pi 


def gaussian_fft_mask_hexagonal(f, pixel_size, lattice_constant=2.46):
    min_distance = pixel_distance_to_first_spot(min(f.shape), pixel_size, lattice_constant) / 6 / 2
    
    coordinates = skimage.feature.peak_local_max(f, min_distance=int(min_distance))
    
    coordinates = coordinates[1:7]
    
    delta_mask = np.zeros_like(f, dtype=float)
    
    delta_mask[coordinates[:,0], coordinates[:,1]] = 1
    
    disk = skimage.morphology.disk(4).astype(float)
    
    mask = scipy.signal.fftconvolve(delta_mask, disk, mode='same')
    
    return mask


def gaussian_fft_filter_hexagonal(image, pixel_size, lattice_constant=2.46):
    
    f = np.fft.fftshift(np.fft.fft2(image))
    
    mask = gaussian_fft_mask_hexagonal(np.abs(f), pixel_size, lattice_constant) 
    
    filtered = np.fft.ifft2(np.fft.ifftshift(image) * np.fft.ifftshift(mask))
    
    return filtered