import pandas as pd
df = pd.read_excel("artifacts/injected_manifest.xlsx", sheet_name="scene_manifest")
s = df[df["scene_type"]=="单列车短时晚点"]["events"]
print(s.describe())
print("1~3占比:", (s.between(1,3)).mean())
