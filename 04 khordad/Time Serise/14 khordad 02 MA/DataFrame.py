import pandas as pd

def load_data(path):
    # اگر جداکننده فایل tab بود: sep='\t' اضافه کن
    df = pd.read_csv(path,delimiter="\t", header=None)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[['date', 'close']]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def train_test_split(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test
