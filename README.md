# Importing Libraries
import pandas as pd, numpy as np, os, json, sys, time, warnings
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Loading the file
df = pd.read_excel('/content/Merged_Actual_and_Forecasted_Electricity_Data.xlsx')

# Column information (non-null count and data type)
print(df.info())
print('='*75)

# Describes the data
print(df.describe())
print('='*75)

# Display few rows
print(df.head())
print('='*75)

# Rows and Columns
print('Shape:',df.shape)
print('='*75)

# Missing Values
print('Missing Values\n')
print(df.isnull().sum())
print('='*75)

#Total Null values in Data Frame
Total_Null_Count = df.isnull().sum().sum()
print('Total Null Count:', Total_Null_Count)
print('='*75)

# Duplicate rows
print('Duplicated Values:',df.duplicated().sum())
print('='*75)

# Ensure output directories exist
os.makedirs('/content/results', exist_ok=True)
os.makedirs('/content/figures', exist_ok=True)

def save_csv(df, path):
  # index is named 'datetime' in every DataFrame, so it is written as the first column
  df.to_csv(path)

def load_csv(path):
  # round_trip keeps float values identical to what was saved
  return pd.read_csv(path, index_col='datetime', parse_dates=True,
                     float_precision='round_trip')

# Read from Colab environment Excel file
df = pd.read_excel('/content/Merged_Actual_and_Forecasted_Electricity_Data.xlsx')
df['Start date']= pd.to_datetime(df['Start date'])
df = df.sort_values('Start date').reset_index(drop=True)
df = df.drop_duplicates(subset='Start date').set_index('Start date')
# Reindex to strict hourly frequency, interpolate small gaps
full_idx= pd.date_range(df.index.min(), df.index.max(), freq='h')
df = df.reindex(full_idx)
df.index.name = 'datetime'
rename = {
    'Actual Consumption - grid load [MWh] Calculated resolutions':'load_actual',
    'Actual Consumption - Residual load [MWh] Calculated resolutions':'residual_load_actual',
    'Actual Generation - Wind offshore [MWh] Calculated resolutions':'wind_offshore_actual',
    'Actual Generation - Wind onshore [MWh] Calculated resolutions':'wind_onshore_actual',
    'Actual Generation - Photovoltaics [MWh] Calculated resolutions':'solar_actual',
    'Actual Generation - Nuclear [MWh] Calculated resolutions':'nuclear_actual',
    'Actual Generation - Lignite [MWh] Calculated resolutions':'lignite_actual',
    'Actual Generation - Hard coal [MWh] Calculated resolutions':'hardcoal_actual',
    'Actual Generation - Fossil gas [MWh] Calculated resolutions':'gas_actual',
    'Actual Generation - Hydropower [MWh] Calculated resolutions':'hydro_actual',
    'Actual Generation - Biomass [MWh] Calculated resolutions':'biomass_actual',
    'Day-Ahead Price - Germany[€/MWh] Calculated resolutions':'price',
    'Forecasted Consumption - grid load [MWh] Calculated resolutions':'load_fc',
    'Forecasted Consumption - Residual load [MWh] Calculated resolutions':'residual_load_fc',
    'Forecasted Generation Day-Ahead - Total [MWh] Calculated resolutions':'gen_total_fc',
    'Forecasted Generation Day-Ahead - Photovoltaics and wind [MWh] Calculated resolutions':'pv_wind_fc',
    'Forecasted Generation Day-Ahead - Wind offshore [MWh] Calculated resolutions':'wind_offshore_fc',
    'Forecasted Generation Day-Ahead - Wind onshore [MWh] Calculated resolutions':'wind_onshore_fc',
    'Forecasted Generation Day-Ahead - Photovoltaics [MWh] Calculated resolutions':'solar_fc',
    'Forecasted Generation Day-Ahead - Other [MWh] Calculated resolutions':'other_fc',
}
df = df.rename(columns=rename)[list(rename.values())]
# interpolate short gaps (<=3h) linearly; leave longer gaps
df = df.interpolate(method='linear', limit=3, limit_direction='both')
print("Shape:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())
print("NaN remaining (price):", df['price'].isna().sum())
print(df.isna().mean().sort_values(ascending=False).head(10))
save_csv(df, '/content/clean_hourly.csv')
df.to_parquet('/content/clean_hourly.parquet')

########
df = load_csv('/content/clean_hourly.csv')
# Restrict working window: 2017-01-01 onward
df = df[df.index >= '2017-01-01']
df = df.drop(columns=['nuclear_actual'])
df['renew_fc']= df['wind_onshore_fc'] + df['wind_offshore_fc'] + df['solar_fc']
df['renew_actual']= df['wind_onshore_actual'] + df['wind_offshore_actual'] + df['solar_actual']
df['renew_share_fc']= (df['renew_fc'] / df['load_fc']).clip(0, 3)
df['thermal_gap_fc']= df['load_fc']- df['renew_fc']
df['hour']= df.index.hour
df['dow']= df.index.dayofweek
df['month']= df.index.month
df['is_weekend']= (df['dow'] >= 5).astype(int)
df['doy']= df.index.dayofyear
# cyclical encodings
df['hour_sin']= np.sin(2*np.pi*df['hour']/24)
df['hour_cos']= np.cos(2*np.pi*df['hour']/24)
df['dow_sin']= np.sin(2*np.pi*df['dow']/7)
df['dow_cos']= np.cos(2*np.pi*df['dow']/7)
df['month_sin']= np.sin(2*np.pi*df['month']/12)
df['month_cos']= np.cos(2*np.pi*df['month']/12)
# price lag / rolling features
for lag in [24, 48, 72, 168]:
  df[f'price_lag_{lag}']= df['price'].shift(lag)
df['price_roll_mean_24']= df['price'].shift(24).rolling(24).mean()
df['price_roll_std_24']= df['price'].shift(24).rolling(24).std()
df['price_roll_mean_168']= df['price'].shift(24).rolling(168).mean()
# renewable forecast lag context
df['renew_fc_lag24']= df['renew_fc'].shift(24)
df = df.dropna().copy()
print(df.shape)
print(df.index.min(), df.index.max())
save_csv(df, '/content/features.csv')
print(df.columns.tolist())

#########
df = load_csv('/content/features.csv')
# Adjust test split to match the end of actual data (e.g., 2025-10-03)
TEST_START= '2025-10-03'
train = df[df.index < TEST_START]
test = df[df.index >= TEST_START]
print("Train:", train.shape, train.index.min(), train.index.max())
print("Test :", test.shape, test.index.min(), test.index.max())
save_csv(train, '/content/train.csv')
save_csv(test, '/content/test.csv')

# Metrics
def compute_metrics(y_true, y_pred):
  y_true= np.asarray(y_true, dtype=float)
  y_pred= np.asarray(y_pred, dtype=float)
  err = y_true- y_pred
  max_val = np.max(np.abs(y_true))
  mae = np.mean(np.abs(err))
  rmse = np.sqrt(np.mean(err**2))
  # sMAPE (robust to near-zero/negative prices, unlike MAPE)
  denom = (np.abs(y_true) + np.abs(y_pred))
  denom = np.where(denom== 0, 1e-6, denom)
  smape = 100 * np.mean(2*np.abs(err) / denom)
  # MAPE restricted to |y_true| > 10 EUR/MWh
  mask = np.abs(y_true) > 10
  mape = 100 * np.mean(np.abs(err[mask] / y_true[mask])) if mask.sum() > 0 else np.nan
  r2 = 1- np.sum(err**2) / np.sum((y_true- np.mean(y_true))**2)
  bias = np.mean(err)
  return dict(MAE=mae, RMSE=rmse, sMAPE=smape, MAPE=mape, R2=r2, Bias=bias)

# ML Models
# Naive
sys.path.insert(0, '/content')
test = load_csv('/content/test.csv')
y_true= test['price'].values
y_pred= test['price_lag_24'].values
m = compute_metrics(y_true, y_pred)
print("Naive persistence (t-24h):", m)
pred_df= pd.DataFrame({'datetime': test.index, 'y_true': y_true, 'naive': y_pred}).set_index('datetime')
save_csv(pred_df, '/content/results/preds_naive.csv')
with open('/content/results/metrics.json', 'w') as f:
  json.dump({'Naive Persistence': m}, f, indent=2)

# XGB/LGBM
sys.path.insert(0, '/content')
train = load_csv('/content/train.csv')
test = load_csv('/content/test.csv')
FEATURES = [c for c in train.columns if c not in ['price']]
Xtr, ytr= train[FEATURES], train['price']
Xte, yte= test[FEATURES], test['price']
with open('/content/results/metrics.json') as f:
  all_metrics= json.load(f)
# ---------------- XGBoost ----------------
t0 = time.time()
xgb_model= xgb.XGBRegressor(
    n_estimators=800, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective='reg:squarederror', random_state=42, n_jobs=4,
    early_stopping_rounds=50, eval_metric='mae'
)

cut = int(len(Xtr)*0.9)
xgb_model.fit(Xtr.iloc[:cut], ytr.iloc[:cut],
              eval_set=[(Xtr.iloc[cut:], ytr.iloc[cut:])], verbose=False)
xgb_pred= xgb_model.predict(Xte)
m = compute_metrics(yte.values, xgb_pred)
print("XGBoost:", m, f"({time.time()-t0:.1f}s, best_iter={xgb_model.best_iteration})")
all_metrics['XGBoost']= m
xgb_imp= pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
xgb_imp.to_csv('/content/results/xgb_importance.csv')
# ---------------- LightGBM ----------------
t0 = time.time()
lgb_train= lgb.Dataset(Xtr.iloc[:cut], ytr.iloc[:cut])
lgb_val= lgb.Dataset(Xtr.iloc[cut:], ytr.iloc[cut:], reference=lgb_train)
params = dict(objective='regression', metric='mae', learning_rate=0.03,
              num_leaves=63, max_depth=7, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=42)
lgb_model= lgb.train(params, lgb_train, num_boost_round=1500, valid_sets=[lgb_val],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
lgb_pred= lgb_model.predict(Xte, num_iteration=lgb_model.best_iteration)
m = compute_metrics(yte.values, lgb_pred)
print("LightGBM:", m, f"({time.time()-t0:.1f}s, best_iter={lgb_model.best_iteration})")
all_metrics['LightGBM']= m
lgb_imp= pd.Series(lgb_model.feature_importance(importance_type='gain'), index=FEATURES).sort_values(ascending=False)
lgb_imp.to_csv('/content/results/lgb_importance.csv')
pred_df= pd.DataFrame({'datetime': test.index, 'y_true': yte.values,
                       'xgboost': xgb_pred, 'lightgbm': lgb_pred}).set_index('datetime')
save_csv(pred_df, '/content/results/preds_gbm.csv')
with open('/content/results/metrics.json', 'w') as f:
  json.dump(all_metrics, f, indent=2)
xgb_model.save_model('/content/results/xgb_model.json')
lgb_model.save_model('/content/results/lgb_model.txt')
print("Saved.")

# SARIMAX
warnings.filterwarnings('ignore')
sys.path.insert(0, '/content')
train = load_csv('/content/train.csv')
test = load_csv('/content/test.csv')
EXOG_COLS= ['thermal_gap_fc', 'renew_fc', 'load_fc']
hist = pd.concat([train, test])
FIT_WINDOW_DAYS= 90
t0 = time.time()
preds= []
actuals = []
idx_out= []
test_dates= test.index
block_starts= pd.date_range(test_dates.min().normalize(), test_dates.max().normalize(), freq='7D')
for bi, bstart in enumerate(block_starts):
  bend = bstart + pd.Timedelta(days=7)
  block = test[(test.index >= bstart) & (test.index < bend)]
  if len(block)== 0:
    continue
  fit_end= bstart
  fit_start= fit_end- pd.Timedelta(days=FIT_WINDOW_DAYS)
  fit_data= hist[(hist.index >= fit_start) & (hist.index < fit_end)]
  y = fit_data['price']
  X = fit_data[EXOG_COLS]
  model = SARIMAX(y, exog=X, order=(2,0,1), seasonal_order=(1,0,1,24),
                  enforce_stationarity=False, enforce_invertibility=False)
  res = model.fit(disp=False, maxiter=40, method='lbfgs')
  steps= len(block)
  fc = res.get_forecast(steps=steps, exog=block[EXOG_COLS])
  yhat= fc.predicted_mean.values
  preds.append(yhat); actuals.append(block['price'].values); idx_out.append(block.index)
  print(f"block {bi+1}/{len(block_starts)} {bstart.date()} done({time.time()-t0:.0f}s elapsed)")
preds= np.concatenate(preds)
actuals = np.concatenate(actuals)
idx_out= idx_out[0].append(idx_out[1:]) if len(idx_out) > 1 else idx_out[0]
m = compute_metrics(actuals, preds)
print("SARIMAX:", m, f"total {time.time()-t0:.1f}s")
with open('/content/results/metrics.json') as f:
  all_metrics= json.load(f)
all_metrics['SARIMAX']= m
with open('/content/results/metrics.json', 'w') as f:
  json.dump(all_metrics, f, indent=2)
pred_df= pd.DataFrame({'datetime': idx_out, 'y_true': actuals, 'sarimax': preds}).set_index('datetime')
save_csv(pred_df, '/content/results/preds_sarimax.csv')
print("Saved.")

# LSTM/GRU
sys.path.insert(0, '/content')
torch.manual_seed(42)
np.random.seed(42)
train = load_csv('/content/train.csv')
test = load_csv('/content/test.csv')
SEQ_FEATURES= ['price','thermal_gap_fc','renew_fc','load_fc',
               'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos']
LOOKBACK = 48
train_dl= train[train.index >= (train.index.max()- pd.Timedelta(days=3*365))]
print("DL train window:", train_dl.shape)
mu = train_dl[SEQ_FEATURES].mean()
sigma= train_dl[SEQ_FEATURES].std()
def make_sequences(df_source, context_df, features, lookback, mu, sigma):
  full = pd.concat([context_df.tail(lookback), df_source])
  scaled = (full[features]- mu) / sigma
  arr = scaled.values.astype(np.float32)
  n = len(df_source)
  X = np.zeros((n, lookback, len(features)), dtype=np.float32)
  for i in range(n):
    X[i]= arr[i:i+lookback]
  y = df_source['price'].values.astype(np.float32)
  return X, y
t0 = time.time()
pre_context= train[train.index < train_dl.index.min()]
Xtr, ytr= make_sequences(train_dl, pre_context, SEQ_FEATURES, LOOKBACK, mu, sigma)
Xte, yte= make_sequences(test, train_dl, SEQ_FEATURES, LOOKBACK, mu, sigma)
print("Sequence build time:", time.time()-t0, "Xtr", Xtr.shape, "Xte", Xte.shape)
Xtr_t= torch.from_numpy(Xtr); ytr_t= torch.from_numpy(ytr)
Xte_t= torch.from_numpy(Xte); yte_t= torch.from_numpy(yte)
n = len(Xtr_t)
cut = int(n*0.9)
train_ds= TensorDataset(Xtr_t[:cut], ytr_t[:cut])
val_ds= TensorDataset(Xtr_t[cut:], ytr_t[cut:])
train_loader= DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader= DataLoader(val_ds, batch_size=512, shuffle=False)
class RNNRegressor(nn.Module):
  def __init__(self, n_features, hidden=48, cell='lstm', layers=1):
    super().__init__()
    rnn_cls= nn.LSTM if cell=='lstm' else nn.GRU
    self.rnn = rnn_cls(n_features, hidden, num_layers=layers, batch_first=True)
    self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32,1))
  def forward(self, x):
    out, _= self.rnn(x)
    return self.head(out[:,-1, :]).squeeze(-1)
def train_model(cell, epochs=8, lr=1e-3, hidden=48):
  model = RNNRegressor(len(SEQ_FEATURES), hidden=hidden, cell=cell)
  opt= torch.optim.Adam(model.parameters(), lr=lr)
  lossf = nn.SmoothL1Loss()
  best_val= np.inf; best_state= None
  hist = []
  for ep in range(epochs):
    model.train()
    tr_loss= 0
    for xb, yb in train_loader:
      opt.zero_grad()
      pred= model(xb)
      loss = lossf(pred, yb)
      loss.backward()
      opt.step()
      tr_loss += loss.item()*len(xb)
    tr_loss /= len(train_ds)
    model.eval()
    val_loss= 0
    with torch.no_grad():
      for xb, yb in val_loader:
        pred= model(xb)
        val_loss += lossf(pred, yb).item()*len(xb)
    val_loss /= len(val_ds)
    hist.append((tr_loss, val_loss))
    if val_loss < best_val:
      best_val= val_loss
      best_state= {k: v.clone() for k, v in model.state_dict().items()}
    print(f" [{cell}] epoch {ep+1}/{epochs} train_loss={tr_loss:.3f} val_loss={val_loss:.3f}")
  model.load_state_dict(best_state)
  return model, hist
with open('/content/results/metrics.json') as f:
  all_metrics= json.load(f)
results_pred= {'datetime': test.index, 'y_true': yte}
for cell in ['lstm', 'gru']:
  t0 = time.time()
  model, hist= train_model(cell, epochs=8, hidden=48)
  model.eval()
  with torch.no_grad():
    pred= model(Xte_t).numpy()
  m = compute_metrics(yte, pred)
  print(cell.upper(), m, f"({time.time()-t0:.1f}s)")
  all_metrics[cell.upper() if cell=='gru' else 'LSTM']= m
  results_pred[cell]= pred
  torch.save(model.state_dict(), f'/content/results/{cell}_model.pt')
pred_df= pd.DataFrame(results_pred).set_index('datetime')
save_csv(pred_df, '/content/results/preds_rnn.csv')
with open('/content/results/metrics.json', 'w') as f:
  json.dump(all_metrics, f, indent=2)
print("Saved.")

# TFT
sys.path.insert(0, '/content')
torch.manual_seed(42)
np.random.seed(42)
train = load_csv('/content/train.csv')
test = load_csv('/content/test.csv')
SEQ_FEATURES= ['price','thermal_gap_fc','renew_fc','load_fc','residual_load_fc',
               'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','is_weekend']
LOOKBACK = 48
train_dl= train[train.index >= (train.index.max()- pd.Timedelta(days=3*365))]
mu = train_dl[SEQ_FEATURES].mean()
sigma= train_dl[SEQ_FEATURES].std()
def make_sequences(df_source, context_df, features, lookback, mu, sigma):
  full = pd.concat([context_df.tail(lookback), df_source])
  scaled = (full[features]- mu) / sigma
  arr = scaled.values.astype(np.float32)
  n = len(df_source)
  X = np.zeros((n, lookback, len(features)), dtype=np.float32)
  for i in range(n):
    X[i]= arr[i:i+lookback]
  y = df_source['price'].values.astype(np.float32)
  return X, y
pre_context= train[train.index < train_dl.index.min()]
Xtr, ytr= make_sequences(train_dl, pre_context, SEQ_FEATURES, LOOKBACK, mu, sigma)
Xte, yte= make_sequences(test, train_dl, SEQ_FEATURES, LOOKBACK, mu, sigma)
print("Xtr", Xtr.shape, "Xte", Xte.shape)
Xtr_t= torch.from_numpy(Xtr); ytr_t= torch.from_numpy(ytr)
Xte_t= torch.from_numpy(Xte); yte_t= torch.from_numpy(yte)
n = len(Xtr_t); cut = int(n*0.9)
train_loader= DataLoader(TensorDataset(Xtr_t[:cut], ytr_t[:cut]), batch_size=256, shuffle=True)
val_loader= DataLoader(TensorDataset(Xtr_t[cut:], ytr_t[cut:]), batch_size=512, shuffle=False)
class GRN(nn.Module):
  def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
    super().__init__()
    self.fc1 = nn.Linear(d_in, d_hidden)
    self.elu = nn.ELU()
    self.fc2 = nn.Linear(d_hidden, d_out)
    self.gate= nn.Linear(d_out, d_out*2)
    self.skip= nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
    self.norm = nn.LayerNorm(d_out)
    self.drop= nn.Dropout(dropout)
  def forward(self, x):
    h = self.fc2(self.elu(self.fc1(x)))
    h = self.drop(h)
    g = self.gate(h)
    a, b= g.chunk(2, dim=-1)
    h = a * torch.sigmoid(b)
    return self.norm(h + self.skip(x))
class VariableSelection(nn.Module):
  def __init__(self, n_features, d_model):
    super().__init__()
    self.feature_grns= nn.ModuleList([GRN(1, d_model//2, d_model) for _ in range(n_features)])
    self.weight_net= nn.Sequential(nn.Linear(n_features*d_model, d_model), nn.ELU(),
                                   nn.Linear(d_model, n_features), nn.Softmax(dim=-1))
  def forward(self, x):
    B, T, F= x.shape
    embs = [grn(x[..., i:i+1]) for i, grn in enumerate(self.feature_grns)]
    stacked = torch.stack(embs, dim=2)
    flat = stacked.reshape(B, T,-1)
    weights= self.weight_net(flat).unsqueeze(-1)
    combined = (stacked * weights).sum(dim=2)
    return combined
class TFTLite(nn.Module):
  def __init__(self, n_features, d_model=32, n_heads=4, dropout=0.1):
    super().__init__()
    self.vsn = VariableSelection(n_features, d_model)
    self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
    self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
    self.attn_norm= nn.LayerNorm(d_model)
    self.grn_out= GRN(d_model, d_model, d_model, dropout=dropout)
    self.head = nn.Linear(d_model, 1)
  def forward(self, x):
    v = self.vsn(x)
    enc, _ = self.lstm(v)
    attn_out, attn_w= self.attn(enc, enc, enc)
    h = self.attn_norm(enc + attn_out)
    h = self.grn_out(h)
    out = self.head(h[:,-1, :]).squeeze(-1)
    return out, attn_w
model = TFTLite(len(SEQ_FEATURES), d_model=32, n_heads=4)
n_params = sum(p.numel() for p in model.parameters())
print("TFT-Lite params:", n_params)
opt= torch.optim.Adam(model.parameters(), lr=2e-3)
lossf = nn.SmoothL1Loss()
epochs= 10
best_val= np.inf; best_state=None
t0 = time.time()
for ep in range(epochs):
  model.train(); tr_loss=0
  for xb, yb in train_loader:
    opt.zero_grad()
    pred, _= model(xb)
    loss = lossf(pred, yb)
    loss.backward(); opt.step()
    tr_loss += loss.item()*len(xb)
  tr_loss /= cut
  model.eval(); val_loss=0
  with torch.no_grad():
    for xb, yb in val_loader:
      pred, _= model(xb)
      val_loss += lossf(pred, yb).item()*len(xb)
  val_loss /= (n-cut)
  if val_loss < best_val:
    best_val= val_loss
    best_state= {k:v.clone() for k,v in model.state_dict().items()}
  print(f"epoch {ep+1}/{epochs} train_loss={tr_loss:.3f} val_loss={val_loss:.3f} ({time.time()-t0:.0f}s)")
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
  pred, attn_w= model(Xte_t)
pred= pred.numpy()
m = compute_metrics(yte, pred)
print("TFT-Lite:", m, f"total {time.time()-t0:.1f}s")
with open('/content/results/metrics.json') as f:
  all_metrics= json.load(f)
all_metrics['TFT']= m
with open('/content/results/metrics.json', 'w') as f:
  json.dump(all_metrics, f, indent=2)
pred_df= pd.DataFrame({'datetime': test.index, 'y_true': yte, 'tft': pred}).set_index('datetime')
save_csv(pred_df, '/content/results/preds_tft.csv')
avg_attn= attn_w.mean(dim=0).detach().numpy()
np.save('/content/results/tft_attn.npy', avg_attn)
torch.save(model.state_dict(), '/content/results/tft_model.pt')
print("Saved.")

# EDA
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})
COLOR_PRICE = '#1f4e79'
COLOR_RENEW = '#2e8b57'
COLOR_ACC = '#c0392b'

df = pd.read_parquet('/content/clean_hourly.parquet')
df = df[df.index >= '2017-01-01']

# 1. Full price history
fig, ax = plt.subplots(figsize=(11,4))
daily = df['price'].resample('D').mean()
ax.plot(daily.index, daily.values, color=COLOR_PRICE, lw=0.6)
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.set_title('German Day-Ahead Electricity Price, Daily Mean (2017-2025)')
ax.set_ylabel('EUR/MWh')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.tight_layout()
fig.savefig('/content/figures/fig01_price_history.png')
plt.close(fig)

# 2. Price distribution
fig, ax = plt.subplots(figsize=(7,4.2))
ax.hist(df['price'].clip(-100,400), bins=120, color=COLOR_PRICE, alpha=0.85)
ax.axvline(0, color=COLOR_ACC, lw=1, ls='--', label='0 EUR/MWh')
ax.set_title('Distribution of Hourly Day-Ahead Prices')
ax.set_xlabel('EUR/MWh (clipped at [-100, 400] for display)')
ax.set_ylabel('Frequency (hours)')
ax.legend()
fig.tight_layout()
fig.savefig('/content/figures/fig02_price_distribution.png')
plt.close(fig)

# 3. Renewable share vs price - scatter/hexbin, plus rolling correlation
df['renew_fc'] = df['wind_onshore_fc'] + df['wind_offshore_fc'] + df['solar_fc']
fig, ax = plt.subplots(figsize=(7,5))
hb = ax.hexbin(df['renew_fc']/1000, df['price'].clip(-50,300), gridsize=45, cmap='viridis', mincnt=3)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Hour count')
ax.set_xlabel('Forecasted Wind+Solar Generation (GWh)')
ax.set_ylabel('Day-Ahead Price (EUR/MWh)')
ax.set_title('Forecasted Renewable Generation vs. Day-Ahead Price')
fig.tight_layout()
fig.savefig('/content/figures/fig03_renewable_vs_price.png')
plt.close(fig)

# 4. Seasonal / hourly boxplots
df['hour'] = df.index.hour
df['month'] = df.index.month
fig, axes = plt.subplots(1, 2, figsize=(12,4.3))
df.boxplot(column='price', by='hour', ax=axes[0], showfliers=False,
           patch_artist=True, boxprops=dict(facecolor=COLOR_PRICE, alpha=0.5))
axes[0].set_title('Price by Hour of Day')
axes[0].set_xlabel('Hour'); axes[0].set_ylabel('EUR/MWh')
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df.boxplot(column='price', by='month', ax=axes[1], showfliers=False,
           patch_artist=True, boxprops=dict(facecolor=COLOR_RENEW, alpha=0.5))
axes[1].set_title('Price by Month')
axes[1].set_xlabel('Month'); axes[1].set_ylabel('EUR/MWh')
axes[1].set_xticklabels(month_names)
fig.suptitle('')
fig.tight_layout()
fig.savefig('/content/figures/fig04_seasonality.png')
plt.close(fig)

# 5. Correlation heatmap of key variables
import matplotlib.colors as mcolors
cols = ['price','load_fc','renew_fc','wind_onshore_fc','wind_offshore_fc','solar_fc',
        'residual_load_fc','thermal_gap_fc' if 'thermal_gap_fc' in df.columns else 'renew_fc']
df['thermal_gap_fc'] = df['load_fc'] - df['renew_fc']
cols = ['price','load_fc','renew_fc','wind_onshore_fc','wind_offshore_fc','solar_fc','residual_load_fc','thermal_gap_fc']
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(7.5,6.5))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8,
                 color='white' if abs(corr.iloc[i,j])>0.6 else 'black')
fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson correlation')
ax.set_title('Correlation: Price vs. Forecasted Fundamentals')
fig.tight_layout()
fig.savefig('/content/figures/fig05_correlation.png')
plt.close(fig)

print("EDA figures saved.")
print(corr['price'])

import pandas as pd, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight':'bold',
    'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})
PALETTE = {
    'Naive Persistence': '#95a5a6', 'SARIMAX': '#8e44ad', 'XGBoost': '#e67e22',
    'LightGBM': '#f1c40f', 'LSTM': '#2980b9', 'GRU': '#16a085', 'TFT': '#c0392b'
}

# merge all predictions
# Load original CSV format results
naive = pd.read_csv('results/preds_naive.csv', index_col='datetime', parse_dates=True)
gbm = pd.read_csv('results/preds_gbm.csv', index_col='datetime', parse_dates=True)
sarimax = pd.read_csv('results/preds_sarimax.csv', index_col='datetime', parse_dates=True)
rnn = pd.read_csv('results/preds_rnn.csv', index_col='datetime', parse_dates=True)
tft = pd.read_csv('results/preds_tft.csv', index_col='datetime', parse_dates=True)

full = naive[['y_true','naive']].join(gbm[['xgboost','lightgbm']], how='outer') \
    .join(sarimax[['sarimax']], how='outer') \
    .join(rnn[['lstm','gru']], how='outer') \
    .join(tft[['tft']], how='outer')
full = full.rename(columns={'naive':'Naive Persistence','xgboost':'XGBoost','lightgbm':'LightGBM',
                             'sarimax':'SARIMAX','lstm':'LSTM','gru':'GRU','tft':'TFT'})
full.to_csv('results/all_predictions.csv')
print(full.shape, full.columns.tolist())

with open('results/metrics.json') as f:
    metrics = json.load(f)
MODEL_ORDER = ['Naive Persistence','SARIMAX','XGBoost','LightGBM','LSTM','GRU','TFT']

# ---- Fig 06: metric comparison bar charts ----
fig, axes = plt.subplots(1, 3, figsize=(14,4.3))
for ax, key, ylabel in zip(axes, ['MAE','RMSE','R2'], ['EUR/MWh','EUR/MWh','R²']):
    vals = [metrics[m][key] for m in MODEL_ORDER]
    colors = [PALETTE[m] for m in MODEL_ORDER]
    bars = ax.bar(MODEL_ORDER, vals, color=colors)
    ax.set_title(key)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=40)
    for b, v in zip(bars, vals):
        ax.annotate(f'{v:.2f}', (b.get_x()+b.get_width()/2, v), ha='center',
                    va='bottom' if v>=0 else 'top', fontsize=8)
fig.suptitle('Out-of-Sample Model Comparison (Test: Oct-Dec 2025, n=2160 hours)', y=1.03, fontweight='bold')
fig.tight_layout()
fig.savefig('/content/figures/fig06_metric_comparison.png', bbox_inches='tight')
plt.close(fig)

# ---- Fig 07: actual vs predicted, one representative week ----
week = full.loc['2025-11-10':'2025-11-17']
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(week.index, week['y_true'], color='black', lw=1.8, label='Actual', zorder=5)
for m in ['Naive Persistence','SARIMAX','LightGBM','GRU','TFT']:
    ax.plot(week.index, week[m], lw=1.1, alpha=0.85, color=PALETTE[m], label=m)
ax.set_title('Actual vs. Forecasted Day-Ahead Price — Sample Week (10-17 Nov 2025)')
ax.set_ylabel('EUR/MWh')
ax.legend(ncol=3, fontsize=8.5, loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
fig.tight_layout()
fig.savefig('/content/figures/fig07_forecast_week.png')
plt.close(fig)

# ---- Fig 08: scatter actual vs predicted (best 3 models) ----
fig, axes = plt.subplots(1, 3, figsize=(13,4.3), sharex=True, sharey=True)
for ax, m in zip(axes, ['LightGBM','GRU','TFT']):
    ax.scatter(full['y_true'], full[m], s=6, alpha=0.35, color=PALETTE[m])
    lims = [full['y_true'].min()-10, full['y_true'].max()+10]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_title(m)
    ax.set_xlabel('Actual (EUR/MWh)')
axes[0].set_ylabel('Predicted (EUR/MWh)')
fig.suptitle('Predicted vs. Actual Price', fontweight='bold')
fig.tight_layout()
fig.savefig('/content/figures/fig08_scatter.png')
plt.close(fig)

# ---- Fig 09: residual distribution by model ----
fig, ax = plt.subplots(figsize=(9,4.5))
for m in MODEL_ORDER:
    resid = full['y_true'] - full[m]
    ax.hist(resid.clip(-100,100), bins=60, histtype='step', lw=1.6, color=PALETTE[m], label=m, density=True)
ax.axvline(0, color='grey', lw=1, ls='--')
ax.set_title('Forecast Error Distribution by Model')
ax.set_xlabel('Error: Actual - Predicted (EUR/MWh)')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('/content/figures/fig09_residual_dist.png')
plt.close(fig)

# ---- Fig 10: error by hour of day (MAE heatmap-ish line) ----
full['hour'] = full.index.hour
fig, ax = plt.subplots(figsize=(10,4.5))
for m in ['Naive Persistence','XGBoost','GRU','TFT']:
    mae_by_hour = (full['y_true']-full[m]).abs().groupby(full['hour']).mean()
    ax.plot(mae_by_hour.index, mae_by_hour.values, marker='o', ms=3, color=PALETTE[m], label=m)
ax.set_title('Mean Absolute Error by Hour of Day')
ax.set_xlabel('Hour'); ax.set_ylabel('MAE (EUR/MWh)')
ax.set_xticks(range(0,24,2))
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig('/content/figures/fig10_error_by_hour.png')
plt.close(fig)

# ---- Fig 11: feature importance (LightGBM) ----
imp = pd.read_csv('results/lgb_importance.csv', index_col=0).squeeze('columns')
imp = imp.sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(8,6))
ax.barh(imp.index, imp.values, color='#f1c40f', edgecolor='#b7950b')
ax.set_title('LightGBM Feature Importance (Gain, Top 15)')
ax.set_xlabel('Total Gain')
fig.tight_layout()
fig.savefig('/content/figures/fig11_feature_importance.png')
plt.close(fig)

# ---- Fig 12: TFT attention heatmap ----
attn = np.load('results/tft_attn.npy')
fig, ax = plt.subplots(figsize=(6.5,5.5))
im = ax.imshow(attn, cmap='magma', aspect='auto')
ax.set_xlabel('Key time step (hours into 48h window)')
ax.set_ylabel('Query time step (hours into 48h window)')
ax.set_title('TFT-Lite Self-Attention Pattern\n(averaged over test batch)')
fig.colorbar(im, ax=ax, label='Attention weight')
fig.tight_layout()
fig.savefig('/content/figures/fig12_tft_attention.png')
plt.close(fig)

# ---- Fig 13: performance during price spikes (top decile actual price) ----
thresh = full['y_true'].quantile(0.9)
spikes = full[full['y_true'] >= thresh]
fig, ax = plt.subplots(figsize=(8,4.5))
maes = [(spikes['y_true']-spikes[m]).abs().mean() for m in MODEL_ORDER]
bars = ax.bar(MODEL_ORDER, maes, color=[PALETTE[m] for m in MODEL_ORDER])
ax.set_title(f'MAE During Price-Spike Hours (top decile, actual > {thresh:.0f} EUR/MWh, n={len(spikes)})')
ax.set_ylabel('MAE (EUR/MWh)')
ax.tick_params(axis='x', rotation=40)
for b,v in zip(bars,maes):
    ax.annotate(f'{v:.1f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom', fontsize=8)
fig.tight_layout()
fig.savefig('/content/figures/fig13_spike_performance.png')
plt.close(fig)

print("All result figures saved.")
print("Spike MAE:", dict(zip(MODEL_ORDER, maes)))

# Additional EDA part 1

import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = '/content'
RES_DIR = f'{BASE_DIR}/results'
PLOT_DIR = f'{RES_DIR}/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})

def load_csv(path):
  return pd.read_csv(path, index_col='datetime', parse_dates=True,
                     float_precision='round_trip')

def savefig(name):
  plt.tight_layout()
  plt.savefig(f'{PLOT_DIR}/{name}.png', bbox_inches='tight')
  plt.show()

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
train = load_csv(f'{BASE_DIR}/train.csv')
test = load_csv(f'{BASE_DIR}/test.csv')

with open(f'{RES_DIR}/metrics.json') as f:
  metrics = pd.DataFrame(json.load(f)).T   # rows = models, cols = metrics

# Merge all prediction files on the datetime index
pred_files = {
    'preds_naive.csv': ['naive'],
    'preds_gbm.csv': ['xgboost', 'lightgbm'],
    'preds_sarimax.csv': ['sarimax'],
    'preds_rnn.csv': ['lstm', 'gru'],
    'preds_tft.csv': ['tft'],
}
preds = None
for fname, cols in pred_files.items():
  path = f'{RES_DIR}/{fname}'
  if not os.path.exists(path):
    print(f"Skipping missing file: {fname}")
    continue
  p = load_csv(path)
  if preds is None:
    preds = p[['y_true'] + cols].copy()
  else:
    preds = preds.join(p[cols], how='left')

MODEL_COLS = [c for c in preds.columns if c != 'y_true']
LABELS = {'naive': 'Naive', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
          'sarimax': 'SARIMAX', 'lstm': 'LSTM', 'gru': 'GRU', 'tft': 'TFT'}
COLORS = {'naive': '#9e9e9e', 'xgboost': '#1f77b4', 'lightgbm': '#2ca02c',
          'sarimax': '#ff7f0e', 'lstm': '#d62728', 'gru': '#9467bd', 'tft': '#8c564b'}

# ------------------------------------------------------------------
# 1. Price history with train/test split
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4))
daily_train = train['price'].resample('D').mean()
daily_test = test['price'].resample('D').mean()
ax.plot(daily_train.index, daily_train.values, color='#1f77b4', lw=0.8, label='Train (daily mean)')
ax.plot(daily_test.index, daily_test.values, color='#d62728', lw=1.2, label='Test (daily mean)')
ax.axvline(test.index.min(), color='k', ls='--', lw=1)
ax.set_title('Day-ahead electricity price (Germany) - train/test split')
ax.set_ylabel('EUR/MWh')
ax.legend()
savefig('01_price_history')

# ------------------------------------------------------------------
# 2. Metric comparison across models
# ------------------------------------------------------------------
name_map = {'Naive Persistence': 'naive', 'XGBoost': 'xgboost', 'LightGBM': 'lightgbm',
            'SARIMAX': 'sarimax', 'LSTM': 'lstm', 'GRU': 'gru', 'TFT': 'tft'}
show_metrics = [m for m in ['MAE', 'RMSE', 'sMAPE', 'R2'] if m in metrics.columns]
fig, axes = plt.subplots(1, len(show_metrics), figsize=(4.2 * len(show_metrics), 4))
axes = np.atleast_1d(axes)
for ax, met in zip(axes, show_metrics):
  vals = metrics[met].sort_values(ascending=(met != 'R2'))
  cols = [COLORS.get(name_map.get(i, ''), '#555555') for i in vals.index]
  ax.barh(vals.index, vals.values, color=cols)
  ax.invert_yaxis()
  ax.set_title(met)
  for y, v in enumerate(vals.values):
    ax.text(v, y, f' {v:.2f}', va='center', fontsize=8)
fig.suptitle('Model comparison on the test set (best on top)', y=1.02)
savefig('02_metric_comparison')

# ------------------------------------------------------------------
# 3. Forecast vs actual - last 7 days of the test set
# ------------------------------------------------------------------
last7 = preds[preds.index >= preds.index.max() - pd.Timedelta(days=7)]
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(last7.index, last7['y_true'], color='black', lw=2, label='Actual')
for c in MODEL_COLS:
  ax.plot(last7.index, last7[c], color=COLORS[c], lw=1, alpha=0.85, label=LABELS[c])
ax.set_title('Forecast vs actual - last 7 days of test set')
ax.set_ylabel('EUR/MWh')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.legend(ncol=4, fontsize=8)
savefig('03_forecast_last_7_days')

# ------------------------------------------------------------------
# 4. Full test period - actual vs best model (by MAE)
# ------------------------------------------------------------------
maes = {c: np.mean(np.abs(preds['y_true'] - preds[c])) for c in MODEL_COLS}
best = min(maes, key=maes.get)
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(preds.index, preds['y_true'], color='black', lw=0.8, label='Actual')
ax.plot(preds.index, preds[best], color=COLORS[best], lw=0.8, alpha=0.9,
        label=f'{LABELS[best]} (best MAE = {maes[best]:.2f})')
ax.set_title('Full test period - actual vs best model')
ax.set_ylabel('EUR/MWh')
ax.legend()
savefig('04_best_model_full_test')

# ------------------------------------------------------------------
# 5. Predicted vs actual scatter for each model
# ------------------------------------------------------------------
n = len(MODEL_COLS)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
axes = np.atleast_1d(axes).ravel()
lo = min(preds['y_true'].min(), preds[MODEL_COLS].min().min())
hi = max(preds['y_true'].max(), preds[MODEL_COLS].max().max())
for ax, c in zip(axes, MODEL_COLS):
  ax.scatter(preds['y_true'], preds[c], s=4, alpha=0.3, color=COLORS[c])
  ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
  ax.set_title(LABELS[c])
  ax.set_xlabel('Actual')
  ax.set_ylabel('Predicted')
for ax in axes[n:]:
  ax.axis('off')
savefig('05_pred_vs_actual_scatter')

# ------------------------------------------------------------------
# 6. Error distribution (boxplot of residuals)
# ------------------------------------------------------------------
errors = pd.DataFrame({LABELS[c]: preds['y_true'] - preds[c] for c in MODEL_COLS})
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.boxplot([errors[c].dropna() for c in errors.columns],
           showfliers=False, patch_artist=True)
ax.set_xticklabels(errors.columns)
ax.axhline(0, color='k', lw=1)
ax.set_title('Forecast error distribution (actual - predicted, outliers hidden)')
ax.set_ylabel('EUR/MWh')
savefig('06_error_boxplot')

# ------------------------------------------------------------------
# 7. MAE by hour of day
# ------------------------------------------------------------------
abs_err = errors.abs()
abs_err['hour'] = abs_err.index.hour
mae_hour = abs_err.groupby('hour').mean()
fig, ax = plt.subplots(figsize=(11, 4.5))
for c in MODEL_COLS:
  ax.plot(mae_hour.index, mae_hour[LABELS[c]], marker='o', ms=3, color=COLORS[c], label=LABELS[c])
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel('Hour of day')
ax.set_ylabel('MAE (EUR/MWh)')
ax.set_title('Mean absolute error by hour of day')
ax.legend(ncol=4, fontsize=8)
savefig('07_mae_by_hour')

# ------------------------------------------------------------------
# 8. Rolling 7-day MAE over the test period
# ------------------------------------------------------------------
roll = abs_err.drop(columns='hour').rolling(24 * 7, min_periods=24).mean()
fig, ax = plt.subplots(figsize=(13, 4))
for c in MODEL_COLS:
  ax.plot(roll.index, roll[LABELS[c]], color=COLORS[c], lw=1.2, label=LABELS[c])
ax.set_title('Rolling 7-day MAE over the test period')
ax.set_ylabel('MAE (EUR/MWh)')
ax.legend(ncol=4, fontsize=8)
savefig('08_rolling_mae')

# ------------------------------------------------------------------
# 9. Feature importance (XGBoost and LightGBM)
# ------------------------------------------------------------------
imp_files = {'XGBoost': 'xgb_importance.csv', 'LightGBM': 'lgb_importance.csv'}
available = {k: v for k, v in imp_files.items() if os.path.exists(f'{RES_DIR}/{v}')}
if available:
  fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 6))
  axes = np.atleast_1d(axes)
  for ax, (name, fname) in zip(axes, available.items()):
    imp = pd.read_csv(f'{RES_DIR}/{fname}', index_col=0).iloc[:, 0].sort_values(ascending=False).head(15)
    ax.barh(imp.index[::-1], imp.values[::-1], color=COLORS[name.lower()])
    ax.set_title(f'{name} - top 15 features')
  savefig('09_feature_importance')

# ------------------------------------------------------------------
# 10. TFT attention heatmap (averaged over test samples)
# ------------------------------------------------------------------
attn_path = f'{RES_DIR}/tft_attn.npy'
if os.path.exists(attn_path):
  attn = np.load(attn_path)
  fig, ax = plt.subplots(figsize=(6.5, 5.5))
  im = ax.imshow(attn, cmap='viridis', aspect='auto')
  ax.set_title('TFT average attention (query step vs key step)')
  ax.set_xlabel('Key position (hours back, 0 = oldest)')
  ax.set_ylabel('Query position')
  ax.grid(False)
  fig.colorbar(im, ax=ax)
  savefig('10_tft_attention')

print(f"\nAll plots saved to {PLOT_DIR}")

# Additional EDA part 2

import os, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RAW = "/Merged_Actual_and_Forecasted_Electricity_Data.xlsx"
FIGS = "/content/ea/"
os.makedirs(FIGS, exist_ok=True)
os.makedirs("/content/eeda", exist_ok=True)

df = pd.read_excel('/content/Merged_Actual_and_Forecasted_Electricity_Data.xlsx', sheet_name="Merged Data")
df["Start date"] = pd.to_datetime(df["Start date"])
df = df.sort_values("Start date").drop_duplicates(subset="Start date").set_index("Start date")

price = "Day-Ahead Price - Germany[’/MWh] Calculated resolutions"
price = "Day-Ahead Price - Germany[’/MWh] Calculated resolutions" if "Day-Ahead Price - Germany[’/MWh] Calculated resolutions" in df.columns else "Day-Ahead Price - Germany[–/MWh] Calculated resolutions"
if price not in df.columns:
    price = [c for c in df.columns if "Day-Ahead Price" in c][0]

rename = {
    price: "price",
    "Actual Consumption - grid load [MWh] Calculated resolutions": "act_load",
    "Actual Generation - Wind offshore [MWh] Calculated resolutions": "act_wind_off",
    "Actual Generation - Wind onshore [MWh] Calculated resolutions": "act_wind_on",
    "Actual Generation - Photovoltaics [MWh] Calculated resolutions": "act_solar",
    "Actual Generation - Biomass [MWh] Calculated resolutions": "act_biomass",
    "Actual Generation - Hydropower [MWh] Calculated resolutions": "act_hydro",
    "Actual Generation - Nuclear [MWh] Calculated resolutions": "act_nuclear",
    "Actual Generation - Lignite [MWh] Calculated resolutions": "act_lignite",
    "Actual Generation - Hard coal [MWh] Calculated resolutions": "act_hardcoal",
    "Actual Generation - Fossil gas [MWh] Calculated resolutions": "act_gas",
    "Forecasted Consumption - grid load [MWh] Calculated resolutions": "fc_load",
    "Forecasted Generation Day-Ahead - Wind onshore [MWh] Calculated resolutions": "fc_wind_on",
    "Forecasted Generation Day-Ahead - Wind offshore [MWh] Calculated resolutions": "fc_wind_off",
    "Forecasted Generation Day-Ahead - Photovoltaics [MWh] Calculated resolutions": "fc_solar",
}
df = df.rename(columns=rename)
df["year"] = df.index.year
df["month"] = df.index.month
df["hour"] = df.index.hour
df["dow"] = df.index.dayofweek

# ---------- summary stats table ----------
summary = df["price"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
summary.to_csv("/content/eeda/price_summary.csv")
print(summary)

neg_by_year = df.groupby("year")["price"].apply(lambda s: (s<0).mean()*100)
print("\n% negative price hours by year:\n", neg_by_year)

renew_share = ((df["act_wind_on"]+df["act_wind_off"]+df["act_solar"]) / df["act_load"]).groupby(df["year"]).mean()*100
print("\nMean renewable share of load by year (%):\n", renew_share)

price_mean_year = df.groupby("year")["price"].mean()
price_std_year = df.groupby("year")["price"].std()
print("\nMean price by year:\n", price_mean_year)
print("\nStd price by year:\n", price_std_year)

# ---------- Figure: yearly boxplot ----------
plt.figure(figsize=(10,5))
years = sorted(df["year"].unique())
data_by_year = [df.loc[df["year"]==y, "price"].clip(-100,300) for y in years]
plt.boxplot(data_by_year, labels=years, showfliers=False)
plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
plt.title("Day-ahead price distribution by year (outliers hidden, clipped to -100..300 –/MWh)")
plt.ylabel("–/MWh");
plt.xticks(rotation=45)
plt.tight_layout();
plt.savefig(FIGS+"eda_price_by_year.png", dpi=150);
plt.close()

# ---------- Figure: renewable share and negative price frequency trend ----------
fig, ax1 = plt.subplots(figsize=(9,5))
ax1.bar(renew_share.index, renew_share.values, color="#66a61e", alpha=0.7, label="Renewable share of load (%)")
ax1.set_ylabel("Mean renewable share of load (%)", color="#66a61e")
ax1.set_xlabel("Year")
ax2 = ax1.twinx()
ax2.plot(neg_by_year.index, neg_by_year.values, color="#d95f02", marker="o", linewidth=2, label="Negative-price hours (%)")
ax2.set_ylabel("Negative-price hours (%)", color="#d95f02")
plt.title("Rising renewable share and rising negative-price frequency, 2015-2025")
fig.tight_layout();
plt.savefig(FIGS+"eda_renew_negprice_trend.png", dpi=150);
plt.close()

# ---------- Figure: generation mix stacked area (yearly mean MW) ----------
mix_cols = ["act_wind_on","act_wind_off","act_solar","act_nuclear","act_lignite","act_hardcoal","act_gas","act_hydro","act_biomass"]
mix_cols = [c for c in mix_cols if c in df.columns]
mix_yearly = df.groupby("year")[mix_cols].mean()
plt.figure(figsize=(10,6))
plt.stackplot(mix_yearly.index, [mix_yearly[c] for c in mix_cols],
              labels=[c.replace("act_", "").replace("_", " ").title() for c in mix_cols],
              colors=["#1b9e77","#66c2a5","#ffd92f","#7570b3","#a6761d","#666666","#e7298a","#377eb8","#8dd3c7"][:len(mix_cols)])
plt.legend(loc="upper left", bbox_to_anchor=(1.01,1), fontsize=8)
plt.ylabel("Mean hourly generation (MWh)")
plt.title("Germany's generation mix, yearly average by fuel, 2015-2025")
plt.tight_layout();
plt.savefig(FIGS+"eda_generation_mix.png", dpi=150);
plt.close()

# ---------- Figure: missingness by column (top offenders) ----------
miss = df.isna().mean().sort_values(ascending=False) * 100
miss = miss[miss > 0.01].head(12)
if not miss.empty:
    plt.figure(figsize=(9,5))
    miss[::-1].plot(kind="barh", color="#e6550d")
    plt.xlabel("% missing")
    plt.title("Columns with non-trivial missingness (raw dataset)")
    plt.tight_layout();
    plt.savefig(FIGS+"eda_missingness.png", dpi=150);
    plt.close()

print("\nSaved EDA figures to", FIGS)
