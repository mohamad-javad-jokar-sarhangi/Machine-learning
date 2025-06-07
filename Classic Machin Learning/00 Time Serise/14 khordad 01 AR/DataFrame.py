import pandas as pd

def load_air_passengers(file_path, test_size=12):
    df = pd.read_csv(file_path, parse_dates=['Month'])
    df = df.rename(columns={'Month': 'date', 'Passengers': 'value'})
    df = df.set_index('date')
    # تقسیم سری به train/test
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test
