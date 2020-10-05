def get_spikes(df, col_name, t_ini=0.3, t_fin=0.1):
    col = df[col_name]
    to_col = col_name + "_bin"
    df[to_col] = 0
    i_pico = -1
    for i in range(len(col)):
        if i_pico < 0:
            if col[i] >= t_ini:
                i_pico = i
        else:
            if col[i] <= t_fin:
                imax = col[i_pico:i].idxmax()
                df.loc[imax, to_col] = 1
                i_pico =-1