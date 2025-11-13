import numpy as np

def crater_reflectance(inc, emi, phase, q, a, f_alpha):
    """Amount of light scattered by the crater.


    Effects of surface roughness on a flat area with a crater.
    A six point gausssian quadrature is used to approximate the
    total amount fo light reflected from the surface.

    The roughness of the surface is simulated by parabolic craters
    that have variable parameters.

    Parameters
    ----------
    inc:
        Incident angle (degree).
    emi:
        Emission angle (degree).
    phase:
        Phase angle (degree).
    q: float
        The depth to radius ratio: ``h / r``
    a: float
        Fraction of radiation singly scattered

    b(lambda) and c(lambda) , the coefficients of the scattering function

    Rhe radius of the crater: r = 25 units
    third implies a 50 x 50 grid for the integration.

    let ``q = h / r``

    assume a scattering law. several forms are:
    * Lambert law: ``I = F * mu_0``

    * Lommel Seeliger law: ``I = F * (mu0 / (mu0 + mu)) * phi(alpha)``
      (our scattering function)

        s(ai,e,alpha) = the amount of light scattered

        b(lambda) = scattering function coefficient
        c(lambda) = scattering function coefficient

        ais = incident angle in crater ( i*)
        es  = emission angle in crater ( e*)

        x = x coordinate in the crater
        y = y coordinate in the crater
        q = h / r

        mu = cos(e)
        mu0 = cos(ai)
        mu_s = cos(es)
        mu0_s = cos(ais)

    ai and e remain constant over the integration
    r remains constant
    x and y are the indexing variables that are also
    the position pointers for the spot on the crater
    all cos, sin and cot functions can be
    calculated beforehand
    all angle mu_st be changed to radians, if they aren't
    already.

    """
    r = 25            # Crater radius
    h = q * r         # Depth of the crater
    r2h = r ** 2 / h

    cos_i, sin_i = np.cos(np.radians(inc)), np.sin(np.radians(inc))
    cos_e, sin_e = np.cos(np.radians(emi)), np.sin(np.radians(emi))

    if sin_e == 0:
        sin_e = 1e-5

    cot_e, cot_i = cos_e / sin_e, cos_i / sin_i

    cos_a = np.clip((np.cos(np.radians(phase)) - cos_i * cos_e) / (sin_i * sin_e), -1, 1)
    sin_a = np.sqrt(1 - cos_a ** 2)

    # (X, Y) value of the center of the crater grid (pixels: -24 to +25)

    x_array = np.linspace(-24.5, 24.5, 50)
    y_array = np.linspace(-24.5, 24.5, 50)

    x_mesh, y_mesh = np.meshgrid(x_array, y_array)

    '''
    # find indices of x,y points where geometry tests are met

    ind = np.argwhere(((x_mesh ** 2 + y_mesh ** 2) < r ** 2) &
                      (((x_mesh - r2h * cot_e) ** 2 + y_mesh ** 2) > r ** 2) &
                      (((x_mesh - r2h * cot_i * cos_a) ** 2 + (y_mesh - r2h * cot_i * sin_a) ** 2) > r ** 2))

    # make list of selected x,y points

    x_sel = np.array([x_mesh[i[0],i[1]] for i in ind])
    y_sel = np.array([y_mesh[i[0],i[1]] for i in ind])

    '''
    
    cond = ((x_mesh ** 2 + y_mesh ** 2) < r ** 2) & \
            (((x_mesh - r2h * cot_e) ** 2 + y_mesh ** 2) > r ** 2) & \
            (((x_mesh - r2h * cot_i * cos_a) ** 2 + (y_mesh - r2h * cot_i * sin_a) ** 2) > r ** 2)
    
    x_sel = x_mesh[np.nonzero(cond)]
    y_sel = y_mesh[np.nonzero(cond)]

    #Calculate the scaterred light.
    x1 = x_sel / r2h
    y1 = y_sel / r2h
    c1 = np.sqrt(1 + 4 * (x1 ** 2 + y1 ** 2))

    # Calculate the scattering angles inside the crater
    mu0_s = np.abs((cos_i - 2 * sin_i * (x1 * cos_a + y1 * sin_a)) / c1)
    mu_s  = (cos_e - 2 * x1 * sin_e) / c1

    # st = b*cos(e)  |   b * cos(e) = b0 *(cos(e) * cos(i)) ** ak
    # (For a minnaert function. For your combination function
    # add up all the scattering angles for that function).
    ref = (f_alpha * a * (mu0_s / (mu_s + mu0_s)) + (1 - a) * mu0_s)

    return np.sum(ref)/1976

def get_A_and_f_alpha(alpha, terrain_type):
    """
    Returns (A, f_alpha) values for a given phase angle and terrain type,.
    based on the work of Dhingra et al. (2021).
    A is clamped between 0 and 1 (albedo). f_alpha is clamped to be >= 0.
    """
    terrain_types = [
        'Bands',
        'Low albedo chaos',
        'Knobby albedo chaos',
        'Mottled albedo chaos',
        'High albedo chaos',
        'Ridged plains',
        'All chaos'
    ]

    A_fit_values = {
        'Bands': (-0.002, 0.868),
        'Low albedo chaos': (-0.004, 0.968),
        'Mottled albedo chaos': (-0.003, 0.905),
        'High albedo chaos': (0.000, 0.402),
        # 'Knobby albedo chaos': no fit; handled elsewhere
        # 'Ridged plains': was "Plains" before
        'Ridged plains': (-0.003, 0.871)
    }

    f_alpha_fit_values = {
        'Bands': (-0.012, 1.499),
        'Low albedo chaos': (-0.036, 2.372),
        'Mottled albedo chaos': (-0.004, 1.081),
        'High albedo chaos': (-0.012, 1.092),
        'Ridged plains': (-0.014, 1.556)
    }

    # Average fits for all chaos units except Knobby (which has no fit)
    if terrain_type == 'All chaos':
        chaos_terrains = ['Low albedo chaos', 'Mottled albedo chaos', 'High albedo chaos']
        slope_A = np.mean([A_fit_values[t][0] for t in chaos_terrains])
        intercept_A = np.mean([A_fit_values[t][1] for t in chaos_terrains])
        slope_f = np.mean([f_alpha_fit_values[t][0] for t in chaos_terrains])
        intercept_f = np.mean([f_alpha_fit_values[t][1] for t in chaos_terrains])

    elif terrain_type in A_fit_values:
        slope_A, intercept_A = A_fit_values[terrain_type]
        slope_f, intercept_f = f_alpha_fit_values[terrain_type]
    else:
        raise ValueError(f"Invalid terrain type '{terrain_type}'. Choose from: {terrain_types}")

    # Raw fits
    A = slope_A * alpha + intercept_A
    f_alpha = slope_f * alpha + intercept_f

    # Clamp to physical bounds
    A = np.clip(A, 0.0, 1.0)
    f_alpha = np.maximum(f_alpha, 0.0)

    return A, f_alpha
