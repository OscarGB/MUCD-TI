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
    return X.groupby(X.index // n).max()

def shanon_entropy(X, wlen):
    p = X.rolling(wlen).apply(lambda x: int(''.join(map(lambda x: str(int(x)), x)), 2)).value_counts()
    p /= p.sum()
    return -np.sum(p*np.log2(p))
    
def shanon_entropy_join(X, wlen):
    p = X.rolling(wlen).apply(lambda x: int(''.join(map(lambda x: str(int(x)), x)), 2))
    p = p.dropna()
    p["joined"] = p.values.tolist()
    p = p.joined.value_counts()
    p /= p.sum()
    return -np.sum(p*np.log2(p))

def mutual_info(X,cols,wlen):
    return shanon_entropy(X[[cols[0]]],wlen) + shanon_entropy(X[[cols[1]]],wlen) - shanon_entropy_join(X[cols],wlen)