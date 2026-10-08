from adriq.Optomechanics import *
window = (1.5,6.5)
raw_data_loaded = load_raw_dict_from_csv("stretch_state_photon.csv")
# 3. Analyze right away with zero online post-selection filtering
stokes, errors, etas, accepted_fraction = perform_tomography_analysis(raw_data_loaded, post_select_window=window)
print("Accepted fraction:", accepted_fraction)
plot_basis_histograms(raw_data_loaded, basis="X", post_select_window=window)
plot_basis_histograms(raw_data_loaded, basis="Y", post_select_window=window)
plot_basis_histograms(raw_data_loaded, basis="Z", post_select_window=window)

print("Stokes vector:", stokes)
print("Errors:", errors)
print("Efficiencies:", etas)

Plot_Stokes_Vector(stokes)
Plot_Density_Matrix(stokes)