import os
import numpy as np
import pandas as pd
from config.settings import TradingConfig


def main():
    path = os.path.join('models_ai', 'base_featured_df.pkl')
    if not os.path.exists(path):
        print(f"Arquivo não encontrado: {path}")
        return
    df = pd.read_pickle(path)
    cols = [c for c in df.columns if 'ema_trend' in c]
    print('ema_trend columns:', cols)
    print('has bare ema_trend:', 'ema_trend' in df.columns)
    primary = TradingConfig().PRIMARY_TIMEFRAME_TRADING
    primary_col = f'ema_trend_{primary}'
    print('PRIMARY_TIMEFRAME_TRADING =', primary)
    print('has ema_trend_primary:', primary_col in df.columns)
    for k in ['tp_regime_up','tp_regime_down','tp_regime_sideways']:
        if k in df.columns:
            s = df[k]
            print(k, 'mean=', float(np.nanmean(s.values)), 'unique_count=', int(s.nunique()))
        else:
            print(k, 'missing')


if __name__ == '__main__':
    main()
