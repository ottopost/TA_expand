import pandas as pd
from ta.momentum import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from ta.others import (
    CumulativeReturnIndicator,
    DailyLogReturnIndicator,
    DailyReturnIndicator,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)


INDIC_PARAMS = {
    'trend_sma_slow': {'window': 26},
    'trend_sma_fast': {'window': 12},
    'trend_ema_slow': {'window': 26},
    'trend_ema_fast': {'window': 12},
    
    'trend_macd': {'window_slow': 26, 'window_fast': 12, 'window_sign': 9},
    'trend_vortex': {'window': 14},
    
    'trend_kst': {'roc1': 10,'roc2': 15,'roc3': 20,'roc4': 30,'window1': 10,'window2': 10,'window3': 10,'window4': 15,'nsig': 9},
    'trend_dpo': {'window': 20},
    'trend_ichimoku': {'window1': 9,'window2': 26,'window3': 52},
    'trend_psar': {'step': 0.02, 'max_step': 0.2},
    
    'trend_adx': {'window': 14},  
    'trend_aroon': {'window': 25},
    'trend_cci': {'window': 20, 'constant': 0.015},
    'trend_visual_ichimoku': {'window1': 9,'window2': 26,'window3': 52},
    
    'momentum_rsi': {'window': 14},
    'momentum_stoch_rsi': {'window': 14, 'smooth1': 3, 'smooth2': 3},
    'momentum_tsi': {'window_slow': 25, 'window_fast': 13},
    'momentum_uo': {'window1': 7, 'window2': 14, 'window3': 28},
    'momentum_stoch': {'window': 14, 'smooth_window': 3},
    'momentum_wr': {'lbp': 14}, 
    'momentum_ao': {'window1': 5, 'window2': 34},
    'momentum_roc': {'window': 12},
    'momentum_ppo': {'window_slow': 26, 'window_fast': 12, 'window_sign': 9},
    'momentum_pvo': {'window_slow': 26, 'window_fast': 12, 'window_sign': 9},
    'momentum_kama': {'window': 10, 'pow1': 2, 'pow2': 30},
    
    'volatility_atr': {'window': 10},
    'volatility_bb': {'window': 20, 'window_dev': 2},
    'volatility_kc': {'window': 10}, 
    'volatility_dc': {'window': 20, 'offset': 0},
    'volatility_ui': {'window': 14},
    
    'volume_adi': {},
    'volume_obv': {},
    'volume_cmf': {},
    'volume_fi': {'window': 13},
    'volume_em': {'window': 14}, 
    'volume_vpt': {},
    'volume_vwap': {'window': 14},
    'volume_mfi': {'window': 14},
    'volume_nvi': {},
    
    'others_dr': {},
    'others_dlr': {},
    'others_cr': {},
}


def add_volume_ta(
    df, high, low, close, volume, fillna, colprefix, vectorized, params
):
    
    if params is None:
        params = {}

    # Accumulation Distribution Index
    if "volume_adi" in params:
        df[f"{colprefix}volume_adi"] = AccDistIndexIndicator(
            high=df[high],
            low=df[low],
            close=df[close],
            volume=df[volume],
            fillna=fillna,
            **params["volume_adi"]
        ).acc_dist_index()

    # On Balance Volume
    if "volume_obv" in params:
        df[f"{colprefix}volume_obv"] = OnBalanceVolumeIndicator(
            close=df[close],
            volume=df[volume],
            fillna=fillna,
            **params["volume_obv"]
        ).on_balance_volume()

    # Chaikin Money Flow
    if "volume_cmf" in params:
        df[f"{colprefix}volume_cmf"] = ChaikinMoneyFlowIndicator(
            high=df[high],
            low=df[low],
            close=df[close],
            volume=df[volume],
            fillna=fillna,
            **params["volume_cmf"]
        ).chaikin_money_flow()

    # Force Index
    if "volume_fi" in params:
        df[f"{colprefix}volume_fi"] = ForceIndexIndicator(
            close=df[close],
            volume=df[volume],
            fillna=fillna,
            **params["volume_fi"]  
        ).force_index()

    # Ease of Movement
    if "volume_em" in params:
        indicator_eom = EaseOfMovementIndicator(
            high=df[high],
            low=df[low],
            volume=df[volume],
            fillna=fillna,
            **params["volume_em"]
        )
        df[f"{colprefix}volume_em"] = indicator_eom.ease_of_movement()
        df[f"{colprefix}volume_sma_em"] = indicator_eom.sma_ease_of_movement()

    # Volume Price Trend
    if "volume_vpt" in params:
        df[f"{colprefix}volume_vpt"] = VolumePriceTrendIndicator(
            close=df[close],
            volume=df[volume],
            fillna=fillna,
            **params["volume_vpt"] 
        ).volume_price_trend()

    # Volume Weighted Average Price
    if "volume_vwap" in params:
        df[f"{colprefix}volume_vwap"] = VolumeWeightedAveragePrice(
            high=df[high],
            low=df[low],
            close=df[close],
            volume=df[volume],
            fillna=fillna,
            **params["volume_vwap"]
        ).volume_weighted_average_price()

    if not vectorized:
        # Money Flow Index 
        if "volume_mfi" in params:
            df[f"{colprefix}volume_mfi"] = MFIIndicator(
                high=df[high],
                low=df[low],
                close=df[close],
                volume=df[volume],
                fillna=fillna,
                **params["volume_mfi"]
            ).money_flow_index()

        # Negative Volume Index
        if "volume_nvi" in params:
            df[f"{colprefix}volume_nvi"] = NegativeVolumeIndexIndicator(
                close=df[close],
                volume=df[volume],
                fillna=fillna,
                **params["volume_nvi"]
            ).negative_volume_index()

    return df


def add_volatility_ta(df, high, low, close, fillna, colprefix, vectorized, params):

    if params is None:
        params = {}

    # Bollinger Bands 
    if "volatility_bb" in params:
        indicator_bb = BollingerBands(
            close=df[close],
            fillna=fillna,
            **params["volatility_bb"]
        )
        df[f"{colprefix}volatility_bbm"] = indicator_bb.bollinger_mavg()
        df[f"{colprefix}volatility_bbh"] = indicator_bb.bollinger_hband()
        df[f"{colprefix}volatility_bbl"] = indicator_bb.bollinger_lband()
        df[f"{colprefix}volatility_bbw"] = indicator_bb.bollinger_wband()
        df[f"{colprefix}volatility_bbp"] = indicator_bb.bollinger_pband()
        df[f"{colprefix}volatility_bbhi"] = indicator_bb.bollinger_hband_indicator()
        df[f"{colprefix}volatility_bbli"] = indicator_bb.bollinger_lband_indicator()

    # Keltner Channel
    if "volatility_kc" in params:    
        indicator_kc = KeltnerChannel(
            close=df[close], 
            high=df[high],
            low=df[low],
            fillna=fillna, 
            **params["volatility_kc"]
        )
        df[f"{colprefix}volatility_kcc"] = indicator_kc.keltner_channel_mband()
        df[f"{colprefix}volatility_kch"] = indicator_kc.keltner_channel_hband()
        df[f"{colprefix}volatility_kcl"] = indicator_kc.keltner_channel_lband()
        df[f"{colprefix}volatility_kcw"] = indicator_kc.keltner_channel_wband()
        df[f"{colprefix}volatility_kcp"] = indicator_kc.keltner_channel_pband()
        df[f"{colprefix}volatility_kchi"] = indicator_kc.keltner_channel_hband_indicator()
        df[f"{colprefix}volatility_kcli"] = indicator_kc.keltner_channel_lband_indicator()

    # Donchian Channel
    if "volatility_dc" in params:
        indicator_dc = DonchianChannel(
            high=df[high],
            low=df[low], 
            close=df[close],
            fillna=fillna,
            **params["volatility_dc"]
        )
        df[f"{colprefix}volatility_dcl"] = indicator_dc.donchian_channel_lband()
        df[f"{colprefix}volatility_dch"] = indicator_dc.donchian_channel_hband()
        df[f"{colprefix}volatility_dcm"] = indicator_dc.donchian_channel_mband()
        df[f"{colprefix}volatility_dcw"] = indicator_dc.donchian_channel_wband()
        df[f"{colprefix}volatility_dcp"] = indicator_dc.donchian_channel_pband()

    if not vectorized:
        # Average True Range
        if "volatility_atr" in params:
            df[f"{colprefix}volatility_atr"] = AverageTrueRange(
                close=df[close], 
                high=df[high],
                low=df[low], 
                fillna=fillna,
                **params["volatility_atr"]
            ).average_true_range()

        # Ulcer Index
        if "volatility_ui" in params:
            df[f"{colprefix}volatility_ui"] = UlcerIndex(
                close=df[close],
                fillna=fillna,
                **params["volatility_ui"]
            ).ulcer_index()

    return df


def add_trend_ta(df, high, low, close, fillna, colprefix, vectorized, params):
    
    if params is None:
        params = {}

    # MACD
    if "trend_macd" in params:
        indicator_macd = MACD(
            close=df[close],
            fillna=fillna,
            **params["trend_macd"]
        )
        df[f"{colprefix}trend_macd"] = indicator_macd.macd()
        df[f"{colprefix}trend_macd_signal"] = indicator_macd.macd_signal()
        df[f"{colprefix}trend_macd_diff"] = indicator_macd.macd_diff()

    # SMAs
    if "trend_sma_fast" in params:
        df[f"{colprefix}trend_sma_fast"] = SMAIndicator(
            close=df[close],  
            fillna=fillna,
            **params["trend_sma_fast"]
        ).sma_indicator()

    if "trend_sma_slow" in params:
        df[f"{colprefix}trend_sma_slow"] = SMAIndicator(
            close=df[close],
            fillna=fillna,
            **params["trend_sma_slow"]
        ).sma_indicator()

    # EMAs
    if "trend_ema_fast" in params:
        df[f"{colprefix}trend_ema_fast"] = EMAIndicator(
            close=df[close],
            fillna=fillna,
            **params["trend_ema_fast"]
        ).ema_indicator()

    if "trend_ema_slow" in params:
        df[f"{colprefix}trend_ema_slow"] = EMAIndicator(
            close=df[close],
            fillna=fillna, 
            **params["trend_ema_slow"]
        ).ema_indicator()

    # Vortex Indicator
    if "trend_vortex" in params:
        indicator_vortex = VortexIndicator(
            high=df[high],
            low=df[low],
            close=df[close],
            fillna=fillna,
            **params["trend_vortex"]  
        )
        df[f"{colprefix}trend_vortex_ind_pos"] = indicator_vortex.vortex_indicator_pos()
        df[f"{colprefix}trend_vortex_ind_neg"] = indicator_vortex.vortex_indicator_neg()
        df[f"{colprefix}trend_vortex_ind_diff"] = indicator_vortex.vortex_indicator_diff()

    # TRIX Indicator
    if "trend_trix" in params:
        df[f"{colprefix}trend_trix"] = TRIXIndicator(
            close=df[close],
            fillna=fillna,
            **params["trend_trix"]
        ).trix()

    # Mass Index
    if "trend_mass_index" in params:
        df[f"{colprefix}trend_mass_index"] = MassIndex(
            high=df[high],
            low=df[low], 
            fillna=fillna,
            **params["trend_mass_index"]
        ).mass_index()

    # DPO Indicator
    if "trend_dpo" in params:
        df[f"{colprefix}trend_dpo"] = DPOIndicator(
            close=df[close],
            fillna=fillna,
            **params["trend_dpo"]
        ).dpo()

    # KST Indicator
    if "trend_kst" in params:
        indicator_kst = KSTIndicator(
            close=df[close],
            fillna=fillna,
            **params["trend_kst"]
        )
        df[f"{colprefix}trend_kst"] = indicator_kst.kst()
        df[f"{colprefix}trend_kst_sig"] = indicator_kst.kst_sig()
        df[f"{colprefix}trend_kst_diff"] = indicator_kst.kst_diff()

    # Ichimoku Indicator
    if "trend_ichimoku" in params:
        indicator_ichi = IchimokuIndicator(
            high=df[high],
            low=df[low],
            fillna=fillna,
            **params["trend_ichimoku"]
        )
        df[f"{colprefix}trend_ichimoku_conv"] = indicator_ichi.ichimoku_conversion_line()
        df[f"{colprefix}trend_ichimoku_base"] = indicator_ichi.ichimoku_base_line()
        df[f"{colprefix}trend_ichimoku_a"] = indicator_ichi.ichimoku_a()
        df[f"{colprefix}trend_ichimoku_b"] = indicator_ichi.ichimoku_b()

    # STC
    if "trend_stc" in params:
        df[f"{colprefix}trend_stc"] = STCIndicator(
            close=df[close],
            fillna=fillna,
            **params["trend_stc"]
        ).stc()

    if not vectorized:
        # ADX
        if "trend_adx" in params:
            indicator_adx = ADXIndicator(
                high=df[high],
                low=df[low],
                close=df[close], 
                fillna=fillna,
                **params["trend_adx"]
            )
            df[f"{colprefix}trend_adx"] = indicator_adx.adx()
            df[f"{colprefix}trend_adx_pos"] = indicator_adx.adx_pos()
            df[f"{colprefix}trend_adx_neg"] = indicator_adx.adx_neg()

        # CCI 
        if "trend_cci" in params:
            df[f"{colprefix}trend_cci"] = CCIIndicator(
                high=df[high],
                low=df[low],
                close=df[close],
                fillna=fillna,
                **params["trend_cci"]
            ).cci()

        # Visual Ichimoku
        if "trend_visual_ichimoku" in params:
            indicator_ichi_visual = IchimokuIndicator(
                high=df[high],
                low=df[low],
                fillna=fillna,
                **params["trend_visual_ichimoku"]
            )
            df[f"{colprefix}trend_visual_ichimoku_a"] = indicator_ichi_visual.ichimoku_a()
            df[f"{colprefix}trend_visual_ichimoku_b"] = indicator_ichi_visual.ichimoku_b()

        # Aroon
        if "trend_aroon" in params:
            indicator_aroon = AroonIndicator(
                close=df[close],
                fillna=fillna,
                **params["trend_aroon"]
            )
            df[f"{colprefix}trend_aroon_up"] = indicator_aroon.aroon_up()
            df[f"{colprefix}trend_aroon_down"] = indicator_aroon.aroon_down()
            df[f"{colprefix}trend_aroon_ind"] = indicator_aroon.aroon_indicator()

        # PSAR 
        if "trend_psar" in params:
            indicator_psar = PSARIndicator(
                high=df[high],
                low=df[low],
                close=df[close],
                fillna=fillna,
                **params["trend_psar"]
            )
            df[f"{colprefix}trend_psar_up"] = indicator_psar.psar_up()
            df[f"{colprefix}trend_psar_down"] = indicator_psar.psar_down()
            df[f"{colprefix}trend_psar_up_indicator"] = indicator_psar.psar_up_indicator()
            df[f"{colprefix}trend_psar_down_indicator"] = indicator_psar.psar_down_indicator()

    return df

def add_momentum_ta(df, high, low, close, volume, fillna, colprefix, vectorized, params):
    
    if params is None:
        params = {}

    # RSI 
    if "momentum_rsi" in params:
        df[f"{colprefix}momentum_rsi"] = RSIIndicator(
            close=df[close],
            fillna=fillna,
            **params["momentum_rsi"]
        ).rsi()

    # Stoch RSI
    if "momentum_stoch_rsi" in params:
        indicator_srsi = StochRSIIndicator(
            close=df[close],
            fillna=fillna,
            **params["momentum_stoch_rsi"]
        )
        df[f"{colprefix}momentum_stoch_rsi"] = indicator_srsi.stochrsi()
        df[f"{colprefix}momentum_stoch_rsi_k"] = indicator_srsi.stochrsi_k()
        df[f"{colprefix}momentum_stoch_rsi_d"] = indicator_srsi.stochrsi_d()

    # TSI
    if "momentum_tsi" in params:
        df[f"{colprefix}momentum_tsi"] = TSIIndicator(
            close=df[close],
            fillna=fillna,
            **params["momentum_tsi"]
        ).tsi()

    if "momentum_uo" in params: 
        df[f"{colprefix}momentum_uo"] = UltimateOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        fillna=fillna,
        **params["momentum_uo"]
        ).ultimate_oscillator()

	# Stochastics 
    if "momentum_stoch" in params:
        indicator_so = StochasticOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        fillna=fillna,
        **params["momentum_stoch"]
        )
        df[f"{colprefix}momentum_stoch"] = indicator_so.stoch()
        df[f"{colprefix}momentum_stoch_signal"] = indicator_so.stoch_signal()

	# Williams %R
    if "momentum_wr" in params:
        df[f"{colprefix}momentum_wr"] = WilliamsRIndicator(
        high=df[high],
        low=df[low], 
        close=df[close],
        fillna=fillna,
        **params["momentum_wr"]
        ).williams_r()

	# Awesome Oscillator
    if "momentum_ao" in params:
        df[f"{colprefix}momentum_ao"] = AwesomeOscillatorIndicator(
        high=df[high],
        low=df[low],
        fillna=fillna,
        **params["momentum_ao"] 
        ).awesome_oscillator()

	# Momentum
    if "momentum_roc" in params:
        df[f"{colprefix}momentum_roc"] = ROCIndicator(
        close=df[close],
        fillna=fillna,
        **params["momentum_roc"]
        ).roc()

	# PPO
    if "momentum_ppo" in params:
        indicator_ppo = PercentagePriceOscillator(
        close=df[close],
        fillna=fillna,
        **params["momentum_ppo"]
        )
        df[f"{colprefix}momentum_ppo"] = indicator_ppo.ppo()
        df[f"{colprefix}momentum_ppo_signal"] = indicator_ppo.ppo_signal()
        df[f"{colprefix}momentum_ppo_hist"] = indicator_ppo.ppo_hist()

    # PVO
    if "momentum_pvo" in params:
        indicator_pvo = PercentageVolumeOscillator(
        volume=df[volume],
        fillna=fillna,
        **params["momentum_pvo"]
        )
        df[f"{colprefix}momentum_pvo"] = indicator_pvo.pvo()
        df[f"{colprefix}momentum_pvo_signal"] = indicator_pvo.pvo_signal()
        df[f"{colprefix}momentum_pvo_hist"] = indicator_pvo.pvo_hist()

    # EMV
    if "volume_emv" in params:
        df[f"{colprefix}volume_emv"] = EMVIndicator(high, 
        low, 
        close, 
        volume, 
        fillna=fillna, 
        **params["volume_emv"]).emv()    


    if not vectorized:
        # KAMA
        if "momentum_kama" in params:
            df[f"{colprefix}momentum_kama"] = KAMAIndicator(
                close=df[close],
                fillna=fillna,
                **params["momentum_kama"]
            ).kama()

    return df
	
def add_others_ta(df, close, fillna, colprefix, params):

    if params is None:
        params = {}

    # Daily Return
    if "others_dr" in params:
        df[f"{colprefix}others_dr"] = DailyReturnIndicator(
            close=df[close],
            fillna=fillna,
            **params["others_dr"]
        ).daily_return()

    # Daily Log Return
    if "others_dlr" in params:
        df[f"{colprefix}others_dlr"] = DailyLogReturnIndicator(
            close=df[close],
            fillna=fillna,  
            **params["others_dlr"]
        ).daily_log_return()

    # Cumulative Return
    if "others_cr" in params:
        df[f"{colprefix}others_cr"] = CumulativeReturnIndicator(
            close=df[close],
            fillna=fillna,
            **params["others_cr"] 
        ).cumulative_return()

    return df


def add_all_ta_features(df, open, high, low, close, volume, fillna, colprefix='', vectorized=False, params=None):
    
    if params is None:
        params = INDIC_PARAMS
        
    df = add_volume_ta(df, high, low, close, volume, fillna, colprefix, vectorized, params)
    df = add_volatility_ta(df, high, low, close, fillna, colprefix, vectorized, params)
    df = add_trend_ta(df, high, low, close, fillna, colprefix, vectorized, params)
    df = add_momentum_ta(df, high, low, close, volume, fillna, colprefix, vectorized, params) 
    df = add_others_ta(df, close, fillna, colprefix, params)
    
    return df
