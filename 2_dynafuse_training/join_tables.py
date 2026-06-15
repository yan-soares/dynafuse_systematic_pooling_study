import pandas as pd
import math
import os
import shutil

main_path_cl = "resultados_consolidados/base"

files_cl_acc = [main_path_cl + '/ACC/' + p for p in os.listdir(main_path_cl + '/ACC/')]
files_cl_devacc = [main_path_cl + '/DEVACC/' + p for p in os.listdir(main_path_cl + '/DEVACC/')]

files_list = [files_cl_acc, files_cl_devacc]

for fl in files_list:
    path_saved = "/".join(fl[0].split('/')[:2])
    filename_saved = "_".join(fl[0].split('/')[1:3])

    dataframes = [
        df.drop(columns=["Unnamed: 0"], errors="ignore")
        for df in (pd.read_csv(file) for file in fl)
    ]
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(path_saved + "/" + filename_saved + ".csv", index=True)

    combined_df = combined_df.map(
        lambda x: f"{x:.2f}".replace(".", ",") if isinstance(x, (float, int)) else x
    )
    combined_df.to_csv(path_saved + "/" + filename_saved + "_google_drive.csv", index=True, sep=";")