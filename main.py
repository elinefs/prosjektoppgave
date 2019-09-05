

def RunModel():
    # ds_folder = ''
    data_folder = '' # Folder with the interpolated data.
    ov_file = '' # Overview file.
    save_folder = '' # Folder to save results.
    save_name = '' # Saved file.

    scan_type = 'T2' # T2 or DWI
    truth_option = 'union' # union, intersection, an or shh
    preprocessing_option = 'None' # None, AutoscalePerScan or AutoscalePerPatient
    method_type = 'LDA' # LDA or QDA
    validation_type = '5-fold' # 5-fold, 10-fold, 20-fold or leave-one-out
    downsample_type = 'None' # None, overall-50/50 ot patient-50/50
    repetition_option = '1' # 1, 5 or 10

    