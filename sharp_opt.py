import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents
from scipy.optimize import minimize
import pandas_datareader.data as web

import matplotlib
matplotlib.use('TkAgg')  # 서버 환경에서 matplotlib 사용 시 필요



def download_prices(tickers, start, end):
    price = yf.download(tickers, start=start, end=end, interval = '1d')['Close'].dropna()
    return price.dropna()


def compute_log_returns(price_df, scale_factor=1, subtract_rf=False, rf_code='DGS10', freq='monthly'):
    log_ret = np.log(price_df).diff().dropna() * scale_factor

    if subtract_rf:
        rf = web.DataReader(rf_code, 'fred', start=log_ret.index[0], end=log_ret.index[-1])
        rf = rf.ffill() / 100 * scale_factor  # 퍼센트 → 소수화
        rf = rf.reindex(log_ret.index).ffill()

        # 무위험수익률 단위 환산
        if freq == 'daily':
            rf_adj = rf / 252
        elif freq == 'weekly':
            rf_adj = rf / 52
        elif freq == 'monthly':
            rf_adj = rf / 12
        else:
            raise ValueError("지원하지 않는 freq입니다. 'daily', 'weekly', 'monthly' 중 선택하세요.")

        log_ret = log_ret.sub(rf_adj.squeeze(), axis=0)

    return log_ret







def estimate_dynamic_mu(
    returns_df: pd.DataFrame,
    level: str = 'local level',
    ar: int | None = None,
    exog_df: pd.DataFrame | None = None,
    mle_regression: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    returns_df : 종목별 수익률 DataFrame (시계열 인덱스)
    exog_df    : 외생변수 DataFrame (returns_df와 같은 인덱스, 여러 컬럼 가능)
    """
    returns_df = returns_df.sort_index()
    # 외생변수도 returns_df 인덱스에 맞춰 보간
    if exog_df is not None:
        exog_df = exog_df.reindex(returns_df.index).ffill()

    filtered_mu  = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    filtered_var = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    results      = {}

    for ticker in returns_df.columns:
        # 1) 해당 종목 시계열과 인덱스 맞추기
        y = returns_df[ticker].dropna()
        exog = exog_df.loc[y.index] if exog_df is not None else None

        # 2) 모델 정의
        model = UnobservedComponents(
            endog=y,
            level=level,
            autoregressive=ar,
            exog=exog,
            mle_regression=mle_regression,
            
        )

        # 3) fit → 내부에서 필터 실행
        res = model.fit(disp=False, maxiter=4000)

        # 4) 상태(level) 및 분산 추출
        mu_series  = pd.Series(res.filtered_state[0],         index=y.index)
        var_series = pd.Series(res.filtered_state_cov[0,0,:], index=y.index)

        filtered_mu.loc[y.index, ticker]  = mu_series
        filtered_var.loc[y.index, ticker] = var_series
        results[ticker] = res

    return filtered_mu, filtered_var, results








def shrink_mean(
    mu_df: pd.DataFrame,
    var_df: pd.DataFrame | None = None,
    lam: float = 0.0,
    method: str = 'manual'
) -> pd.DataFrame:
    """
    mu_df  : T×N 예상 수익(칼만 필터로 추정된 μ)
    var_df : T×N posterior state covariance (P_{i,t|t} 대각 성분)
    lam    : manual 모드에서 쓰는 λ
    method : 'manual' 또는 'auto'
    """
    # ───────────────────────────────────────── manual 모드 ─────
    if method == 'manual':
        mean_s = mu_df.mean(axis=1)
        return mu_df.mul(1 - lam, axis=0).add(mean_s.mul(lam), axis=0)

    # ───────────────────────────────────────── auto 모드 ──────
    if var_df is None:
        raise ValueError("method='auto'일 때는 var_df를 반드시 넘겨주세요.")

    # 인덱스·칼럼 정렬
    var_df = var_df.reindex(mu_df.index)[mu_df.columns]

    T, N = mu_df.shape
    out  = pd.DataFrame(index=mu_df.index, columns=mu_df.columns, dtype=float)

    for t in range(T):
        mu_t   = mu_df.iloc[t]
        mu_bar = mu_t.mean()
    
        # --- 1) 자산별 표준편차 (float 형 보장)
        var_t_series = var_df.iloc[t].astype(float)          # ← 핵심 수정
        sigma = np.sqrt(var_t_series.clip(lower=1e-12))      # 0 방지
    
        # --- 2) 표준화
        z      = (mu_t - mu_bar) / sigma
        z2_sum = max((z ** 2).sum(), 1e-6)
    
        # --- 3) Stein λ
        lam_opt = max(0.0, 1 - (N - 3) / z2_sum)
    
        # --- 4) 복원 후 수축
        out.iloc[t] = mu_bar + lam_opt * (mu_t - mu_bar)


    return out






def constant_correlation_target(S):
    std = np.sqrt(np.diag(S))
    corr = S / np.outer(std, std)
    N = S.shape[0]
    mean_corr = (corr.sum() - N) / (N * (N - 1))
    T = mean_corr * np.outer(std, std)
    np.fill_diagonal(T, std ** 2)
    return T


def subtract_risk_free_from_mu(
    mu_df: pd.DataFrame,
    rf_code: str = 'DGS10',
    freq: str = 'monthly'
) -> pd.DataFrame:
    """
    mu_df      : 기대수익률 추정치 (index는 시계열, columns는 자산)
    rf_code    : FRED에서 가져올 무위험수익률 코드 (예: DGS10)
    freq       : 'daily', 'weekly', 'monthly' 중 하나로, 무위험수익률 환산 기준

    Returns
    -------
    mu_excess_df : 기대 초과수익률 (mu - rf)
    """
    from pandas_datareader.data import DataReader

    # 1. 무위험수익률 다운로드
    rf = DataReader(rf_code, 'fred', start=mu_df.index[0], end=mu_df.index[-1])
    rf = rf.ffill() / 100  # 퍼센트 → 소수점

    # 2. 기간 환산 (연 → 일/월/주)
    if freq == 'daily':
        rf = rf / 252
    elif freq == 'weekly':
        rf = rf / 52
    elif freq == 'monthly':
        rf = rf / 12
    else:
        raise ValueError("지원하지 않는 freq입니다. 'daily', 'weekly', 'monthly' 중 선택하세요.")

    # 3. 시점 정렬 및 결측 처리
    rf = rf.reindex(mu_df.index).ffill()

    # 4. 기대수익률에서 무위험수익률 차감
    mu_excess_df = mu_df.sub(rf.squeeze(), axis=0)

    return mu_excess_df


from mgarch import mgarch
from tqdm import tqdm

def rolling_dcc_garch(returns_df, window=252, step=25, dist='t'):
    cov_list, idx_list = [], []

    T = len(returns_df)
    if window >= T - 1:
        raise ValueError("window는 전체 길이-1보다 작아야 1스텝 앞 예측 라벨링이 가능합니다.")

    # ── 초기 fit: [0, window)로 적합 ──────────────────────────
    model = mgarch(dist=dist)
    model.fit(returns_df.iloc[:window].values)

    # ── 첫 예측: 정보시점=window-1 → 대상일=window 로 라벨 ────  (★오프바이원 수정)
    pred  = model.predict(1)
    cov_t = pd.DataFrame(pred['cov'], index=returns_df.columns, columns=returns_df.columns)
    cov_list.append(cov_t)
    idx_list.append(returns_df.index[window])   # ← 기존: window-1 (잘못)

    # ── 메인 루프: end = window .. T-2 (항상 end+1이 대상일) ───── (★루프 범위 조정)
    for end in tqdm(range(window, T - 1), desc="Rolling DCC-GARCH Daily Fit", unit="day"):
        # 1) 주기적 리핏: 윈도우 [end-window+1, end]로 재적합
        if ((end - window + 1) % step) == 0:
            start = end - window + 1
            sub_data = returns_df.iloc[start:end+1].values
            model = mgarch(dist=dist)
            model.fit(sub_data)
            print(f"[DEBUG] Model refitted on rows {start} ~ {end}")

        # 2) 리핏 사이 ‘일별 업데이트’ 수행 → 없으면 일일 리핏으로 폴백 (★계단현상 제거)
        updated = False
        y_t = returns_df.iloc[end].values
        for attr in ("update_one", "update", "filter_one", "filter"):
            if hasattr(model, attr):
                try:
                    getattr(model, attr)(y_t)
                    updated = True
                    break
                except TypeError:
                    pass
        if not updated and ((end - window + 1) % step) != 0:
            # 폴백: 리핏 없는 날에도 마지막 window로 재적합
            start = end - window + 1
            sub_data = returns_df.iloc[start:end+1].values
            model = mgarch(dist=dist)
            model.fit(sub_data)

        # 3) t+1 대상 예측 생성 및 라벨을 target(day+1)로 기록
        pred  = model.predict(1)
        cov_t = pd.DataFrame(pred['cov'], index=returns_df.columns, columns=returns_df.columns)
        cov_list.append(cov_t)
        idx_list.append(returns_df.index[end + 1])  # ← 기존: end (as-of 라벨)

    return pd.Series(cov_list, index=idx_list, dtype=object)





def optimize_weights(mu, cov, objective='sharpe', ridge=1e-3, sum_to_one=True):
    mu_arr = mu.values
    cov_mat = cov.values
    N = len(mu_arr)

    if objective == 'sharpe':
        def obj(w):
            ret = w @ mu_arr
            vol = np.sqrt(w @ cov_mat @ w)
            ratio = -ret / vol if vol > 0 else np.inf
            penalty = ridge * np.sum(w ** 2)
            return ratio + penalty

    elif objective == 'kelly':
        def obj(w):
            utility = -(w @ mu_arr - 0.5 * w @ cov_mat @ w)
            penalty = ridge * np.sum(w ** 2)
            return utility + penalty

    else:
        raise ValueError("objective는 'sharpe' 또는 'kelly'만 가능합니다.")

    bounds = [(0, 1)] * N if sum_to_one else [(0, None)] * N

    # ✅ 비중합 = 1 제약 여부
    if sum_to_one:
        cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    else:
        cons = ()

    w0 = np.ones(N) / N

    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
    return pd.Series(res.x if res.success else np.full(N, np.nan), index=mu.index)







def rolling_portfolio_weights(mu_df, cov_list, objective='sharpe', ridge=1e-3):
    weights = []
    for t in range(len(mu_df)):
        weights.append(optimize_weights(mu_df.iloc[t], cov_list[t], objective=objective, ridge=ridge))
    return pd.DataFrame(weights, index=mu_df.index)

# ===========================
# 실행 예시
# ===========================
tickers = ['XLE',  # 에너지 (Energy)
           'XLB',  # 소재 (Materials)
           'XLI',  # 산업재 (Industrials)
           'XLP',  # 필수소비재 (Consumer Staples)
           'XLY',  # 자유소비재 (Consumer Discretionary)
           'XLV',  # 헬스케어 (Healthcare)
           'XLF',  # 금융 (Financials)
           'XLK',  # 정보기술 (Information Technology)
           'GDX'   # 금채굴 (Gold Miner)
               ] 



#tickers = ['QQQ','GLD']

prices = download_prices(tickers, '1970-01-01', '2025-12-31')


log_returns = compute_log_returns(prices, subtract_rf=True, freq='daily',rf_code='DGS3MO')

mu_df, var_df, resu = estimate_dynamic_mu(log_returns)



mu_shrunk = shrink_mean(mu_df, var_df=var_df, method='auto')


mu_shrunk.iloc[:25] = np.nan
mu_shrunk= mu_shrunk.dropna()



# 3. 확장 DCC-GARCH 공분산
dcc_cov_series = rolling_dcc_garch(log_returns, window=252, step=25, dist='t')

# 4. mu_shrunk와 날짜 맞춤
mu_for_opt = mu_shrunk.loc[dcc_cov_series.index]

#sharpe_weights = rolling_portfolio_weights(mu_shrunk, cov_series, objective='sharpe', ridge= 0.1)
#kelly_weights = rolling_portfolio_weights(mu_shrunk, cov_series, objective='kelly', ridge= 0.0)

# 5. Kelly / Sharpe 최적화
sharpe_weights = rolling_portfolio_weights(mu_for_opt, list(dcc_cov_series), objective='sharpe', ridge=0.1)


import bt
from bt.algos import Or, RunOnce, RunIfOutOfBounds

# 1. Sharpe 전략 데이터 준비
common_idx = prices.index.intersection(sharpe_weights.index)
prices_bt = prices.loc[common_idx]
weights_bt = sharpe_weights.loc[common_idx]

# ------------------------
# Sharpe 기반 전략 정의
# ------------------------
strategy_sharpe = bt.Strategy(
    'Sharpe Strategy',
    [
        bt.algos.WeighTarget(weights_bt),
        Or([
            bt.algos.RunOnce(),
            bt.algos.RunIfOutOfBounds(0.05)
        ]),
        bt.algos.Rebalance()
    ]
)
test_bt = bt.Backtest(strategy_sharpe, prices_bt)

# ------------------------
# Equal Weight 벤치마크 전략
# ------------------------
benchmark_price = bt.get(tickers, start=prices_bt.index[0], end=prices_bt.index[-1])
common_idx = prices_bt.index.intersection(benchmark_price.index)
prices_bt = prices_bt.loc[common_idx]
benchmark_price = benchmark_price.loc[common_idx]
weights_bt = weights_bt.loc[common_idx]

strategy_equal_weight = bt.Strategy(
    'Equal Weight',
    [
        bt.algos.RunMonthly(),
        bt.algos.SelectAll(),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance()
    ]
)
benchmark_bt = bt.Backtest(strategy_equal_weight, benchmark_price)

# ------------------------
# SPY 벤치마크 전략
# ------------------------
benchmark_price2 = bt.get('SPY', start=prices_bt.index[0], end=prices_bt.index[-1])
common_idx = prices_bt.index.intersection(benchmark_price2.index)
prices_bt = prices_bt.loc[common_idx]
benchmark_price2 = benchmark_price2.loc[common_idx]
weights_bt = weights_bt.loc[common_idx]

strategy_spy = bt.Strategy(
    'SPY',
    [
        bt.algos.RunOnce(),
        bt.algos.SelectAll(),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance()
    ]
)
benchmark_bt2 = bt.Backtest(strategy_spy, benchmark_price2)


# ------------------------
# 실행 및 결과 비교
# ------------------------
result = bt.run(test_bt, benchmark_bt, benchmark_bt2)
result.display()



import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))  
ax = result.plot()
plt.show()
