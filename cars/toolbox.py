def find_missing(df, token='?'):
    col_acc = []
    for c in df.columns:
        if token in df[c].values:
            col_acc.append(c)
    print(*col_acc)


def show_object_columns(df):
    return df.select_dtypes(include=['object'])


def to_category(df, columns):
    for c in columns:
        df[c] = df[c].astype('category')
    return df


def convert_type(df, orig, to='float64'):
    cols = df.select_dtypes(include=[orig]).columns
    for c in cols:
        df[c] = df[c].astype(to)


def show_nans(df):
    return df.isnull().sum()


def learning_rate_gen(start=0.01, iterations=8):
    acc = []
    for i in range(0, iterations):
        if i == 0:
            acc.append(start)
        acc.append(acc[-1] * 2)
    return acc
