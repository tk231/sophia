def calc_ecc_rcc_dual():
    import pandas
    import numpy
    import uncertainties
    import tqdm

    path_to_lookup_table = '/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/results/total_results_correct.xlsx'
    df = pandas.read_excel(path_to_lookup_table)

    new_df = pandas.DataFrame(
        columns=['imgname', 'voltage_max_dfdt', 'calcium_max_dfdt', 'APD50', 'CaTD80', 'ECC', 'RRC'])

    # Find list of all voltage and calcium imgs
    list_of_imgs = []

    for _, row in df.iterrows():
        # Check if either 'slicemean_cal_maxdfdt' or 'slicemean_vol_maxdfdt' contains a value
        if pandas.notna(row['slicemean_vol_maxdfdt']) or pandas.notna(row['slicemean_cal_maxdfdt']):
            list_of_imgs.append(row['imgname'])

    list_of_calcium_layers = []
    list_of_voltage_layers = []

    # Get only calcium layer
    for imgs in list_of_imgs:
        img_basename, img_type = imgs.split('_')
        if img_type == 'calcium':
            list_of_calcium_layers.append(imgs)
        elif img_type == 'voltage':
            list_of_voltage_layers.append(imgs)

    smaller_list = list_of_calcium_layers if len(list_of_calcium_layers) < len(list_of_voltage_layers) else list_of_voltage_layers

    # Get paired details
    for layer in tqdm.tqdm(smaller_list):
        layer_basename, layer_type = layer.split('_')
        if layer_type == 'voltage':
            corr_layer = f"{layer_basename}_calcium"

            voltage_max_dfdt_str = df.loc[df['imgname'] == layer, 'slicemean_vol_maxdfdt']
            APD50_str = df.loc[df['imgname'] == layer, 'slicemean_APD50']
            calcium_max_dfdt_str = df.loc[df['imgname'] == corr_layer, 'slicemean_cal_maxdfdt']
            CaTD80_str = df.loc[df['imgname'] == corr_layer, 'slicemean_CaTD80']

        elif layer_type == 'calcium':
            corr_layer = f"{layer_basename}_voltage"

            voltage_max_dfdt_str = df.loc[df['imgname'] == corr_layer, 'slicemean_vol_maxdfdt'].item()
            APD50_str = df.loc[df['imgname'] == corr_layer, 'slicemean_APD50'].item()
            calcium_max_dfdt_str = df.loc[df['imgname'] == layer, 'slicemean_cal_maxdfdt'].item()
            CaTD80_str = df.loc[df['imgname'] == layer, 'slicemean_CaTD80'].item()

        voltage_max_dfdt_item = voltage_max_dfdt_str.item() if len(voltage_max_dfdt_str) != 0 else numpy.NaN
        APD50_str_item = APD50_str.item() if len(APD50_str) != 0 else numpy.NaN
        calcium_max_dfdt_item = calcium_max_dfdt_str.item() if len(calcium_max_dfdt_str) != 0 else numpy.NaN
        CaTD80_item = CaTD80_str.item() if len(CaTD80_str) != 0 else numpy.NaN


        if not isinstance(voltage_max_dfdt_item, str) or not isinstance(APD50_str_item, str) or not isinstance(calcium_max_dfdt_item, str) or not isinstance(CaTD80_item, str):
            print(f"{layer_basename} has nan values!")
            pass
        else:
            voltage_max_dfdt = uncertainties.ufloat_fromstr(voltage_max_dfdt_item)
            calcium_max_dfdt = uncertainties.ufloat_fromstr(calcium_max_dfdt_item)
            CaTD80 =  uncertainties.ufloat_fromstr(CaTD80_item)
            APD50 =  uncertainties.ufloat_fromstr(APD50_str_item)

            ecc = - voltage_max_dfdt + calcium_max_dfdt
            rrc = ecc + CaTD80 - APD50

            new_row = {'imgname': layer_basename,
                       'voltage_max_dfdt': voltage_max_dfdt,
                       'calcium_max_dfdt': calcium_max_dfdt,
                       'APD50': APD50,
                       'CaTD80': CaTD80,
                       'ECC': ecc,
                       'RRC': rrc}

            new_df = pandas.concat([new_df, pandas.DataFrame([new_row])], ignore_index=True)

    new_df.to_excel('/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/results/ecc_rrc_table_corrected.xlsx', index=False)

calc_ecc_rcc_dual()