import numpy as np
import pandas as pd

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

def resolution(X, n):
    if n == 1:
        return X
    return X.groupby(X.index // n).max()

def shanon_entropy(X, wlen):
    if wlen == 1:
        p = X.value_counts()
    else:
        p = X.rolling(wlen).apply(lambda x: np.packbits(x.astype('u1'))[0], raw=True).value_counts()
    p /= p.sum()
    return -np.sum(p*np.log2(p))
    
def shanon_entropy_join(X, wlen):
    if wlen == 1:
        p = X
    else:
        p = X.rolling(wlen).apply(lambda x: np.packbits(x.astype('u1'))[0], raw=True)
        p = p.dropna()
    
    p["joined"] = p[p.columns[0]] * 1000 + p[p.columns[1]]
    p = p.joined.value_counts()
    p /= p.sum()
    return -np.sum(p*np.log2(p))

def mutual_info(X,cols,wlen):
    entropy1 = shanon_entropy(X[[cols[0]]],wlen)
    print(f"Entropia primera columna: {entropy1}")
    entropy2 = shanon_entropy(X[[cols[1]]],wlen)
    print(f"Entropia segunda columna: {entropy2}")
    joined_entropy = shanon_entropy_join(X[cols],wlen)
    print(f"Entropia conjunta: {joined_entropy}")
    return  entropy1+ entropy2 - joined_entropy

def mutual_info_optimized(X, cols, wlen):
    if wlen == 1:
        p = X
    else:
        p = X.rolling(wlen).apply(lambda x: np.packbits(x.astype('u1'))[0], raw=True)
        p = p.dropna()
    
    p["joined"] = p[cols[0]] * 1000 + p[cols[1]]
    p1 = p[cols[0]].value_counts()
    p1 /= p1.sum()
    entropy1 = -np.sum(p1*np.log2(p1))
    print(f"Entropia primera columna: {entropy1}")
    p2 = p[cols[1]].value_counts()
    p2 /= p2.sum()
    entropy2 = -np.sum(p2*np.log2(p2))
    print(f"Entropia segunda columna: {entropy2}")
    p3 = p['joined'].value_counts()
    p3 /= p3.sum()
    joined_entropy = -np.sum(p3*np.log2(p3))
    print(f"Entropia conjunta: {joined_entropy}")
    del p, p1, p2, p3
    return  entropy1 + entropy2 - joined_entropy

def get_max_window(X, cols):
    windows = []
    for c in cols:
        w = X.index[X[c] > 0]
        windows.append(min(w[1:] - w[:-1]))
    return min(windows)