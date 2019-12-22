import pandas as pd

df = pd.DataFrame()
df["Delay"] = [3]
df["ShootSize"] = [(400,300)]
df["MaxImages"] = [100]
df["BackGround"] = [(255,255,0)]
df.to_csv("setting.csv",index=None)