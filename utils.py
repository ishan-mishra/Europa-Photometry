import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from ssi import SSI
import os

def bin_reflectance(ref, angle_1, angle_2, bin_interval=1, clean_ref=True):
    
    if clean_ref:
        # filter out the nans from the reflectance data, also limit values to 0 and 1

        clean_part = ~np.isnan(ref)
        ref = ref[clean_part]
        angle_1 = angle_1[clean_part]
        angle_2 = angle_2[clean_part]

        physical_part = np.argwhere((ref > 0) & (ref < 1))
        ref = ref[physical_part]
        angle_1 = angle_1[physical_part]
        angle_2 = angle_2[physical_part]        

    if angle_1.min() == 0:
        angle_bin_axis = np.arange(0.01, angle_1.max(),step=bin_interval)
    else:
        angle_bin_axis = np.arange(angle_1.min(), angle_1.max(),step=bin_interval)

    angle_1_new = np.empty(angle_bin_axis.size)
    angle_1_new_error = np.empty_like(angle_1_new)
    angle_2_new = np.empty_like(angle_1_new)
    angle_2_new_error = np.empty_like(angle_1_new)
    ref_new = np.empty_like(angle_1_new)
    ref_new_error = np.empty_like(angle_1_new)

    for k in range(angle_1_new.size):
        ind = np.where((angle_1 > angle_bin_axis[k] - bin_interval) & (angle_1 < angle_bin_axis[k] + bin_interval))
        angle_1_new[k] = np.mean(angle_1[ind])
        angle_1_new_error[k] = np.std(angle_1[ind])
        angle_2_new[k] = np.mean(angle_2[ind])
        angle_2_new_error[k] = np.std(angle_2[ind])
        ref_new[k] = np.mean(ref[ind])
        ref_new_error[k] = np.std(ref[ind])

    # clean up: remove nan or zero-error bins
    clean_part = (
        np.isfinite(ref_new) &
        np.isfinite(ref_new_error) &
        (ref_new_error > 0)
    )
    ref_new = ref_new[clean_part]
    ref_new_error = ref_new_error[clean_part]
    angle_1_new = angle_1_new[clean_part]
    angle_1_new_error = angle_1_new_error[clean_part]
    angle_2_new = angle_2_new[clean_part]
    angle_2_new_error = angle_2_new_error[clean_part]
    

    return angle_1_new, angle_1_new_error, angle_2_new, angle_2_new_error, ref_new, ref_new_error




def q_to_mean_slope(q, q_err):
    '''converts the roughness parameter q to mean slope angle parameter from Hapke's model'''

    theta = np.arctan(4*q/3)*180/np.pi
    theta_err = q_err*12/(16*q**2 + 9)*180/np.pi  # theta_err = q_err*(derivative of arctan(4q/3))*180/np.pi

    return theta, theta_err

def mean_slope_to_q(theta):
    '''converts mean slope value to roughness parameter q'''

    q = 3/4*np.tan(np.pi*theta/180)

    return q


def scatter_ssi_footprints_on_europa(
    ax,
    image_paths,
    europa_map_path='EuropaVoyagerGalileoSSI_MAP2_SIMP.png',
    point_color='cyan',
    point_size=1,
    alpha=0.005
):
    """
    Plots SSI image footprints as dense scatter plots on a Europa basemap.

    Parameters:
    - ax : matplotlib Axes
        The axis to plot on.
    - image_paths : list of str
        Paths to .pho.cub SSI image files.
    - europa_map_path : str, optional
        Path to Europa basemap PNG image.
    - point_color : str, optional
        Color of scatter points (default: 'cyan').
    - point_size : int, optional
        Marker size for scatter points (default: 1).
    - alpha : float, optional
        Transparency for individual points (default: 0.005).
    """

    # Load and display the Europa basemap
    europa_map = plt.imread(europa_map_path)
    europa_map = europa_map[:, ::-1]  # Flip to place 0° on the left (west longitude convention)
    ax.imshow(europa_map, zorder=0, aspect='auto', cmap='gray', extent=(0, 360, -90, 90))

    ax.set_xlabel('West Longitude [°]')
    ax.set_ylabel('Latitude [°]')
    ax.set_title('Galileo SSI Image Footprints on Europa')

    for img_path in image_paths:
        try:
            img = SSI(img_path)
            lat = np.array(img.lat)
            lon = np.array(img.lon)

            valid_mask = np.isfinite(lat) & np.isfinite(lon)
            valid_lat = lat[valid_mask]
            valid_lon = lon[valid_mask]

            ax.scatter(valid_lon, valid_lat, s=point_size, c=point_color, alpha=alpha, zorder=1)

        except Exception as e:
            print(f"[WARNING] Skipped {os.path.basename(img_path)} due to error: {e}")

