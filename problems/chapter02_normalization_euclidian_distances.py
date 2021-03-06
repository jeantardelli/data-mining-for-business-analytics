import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

df = pd.DataFrame({"age": [25, 56, 65, 32, 41, 49],
                   "income": [49000, 156000, 99000, 192000, 39000, 57000]})

print(df)
#   age  income
#0   25   49000
#1   56  156000
#2   65   99000
#3   32  192000
#4   41   39000
#5   49   57000

z_norm = (df - df.mean()) / df.std()
min_max_norm = (df - df.min()) / (df.max() - df.min())

print("Normalization (z-score):\n", z_norm)
#Normalization (z-score):
#         age    income
#0 -1.313253 -0.790027
#1  0.756790  0.911977
#2  1.357770  0.005302
#3 -0.845824  1.484614
#4 -0.244844 -0.949093
#5  0.289361 -0.662774

print("Normalization (min-max):\n", min_max_norm)
#Normalization (min-max):
#      age    income
#0  0.000  0.065359
#1  0.775  0.764706
#2  1.000  0.392157
#3  0.175  1.000000
#4  0.400  0.000000
#5  0.600  0.117647

dist = pd.DataFrame(euclidean_distances(df, df))
print("Euclidian distances without normalization:\n", dist)
#Euclidian distances without normalization:
#               0              1             2              3              4              5
#0       0.000000  107000.004491  50000.016000  143000.000171   10000.012800    8000.036000
#1  107000.004491       0.000000  57000.000711   36000.008000  117000.000962   99000.000247
#2   50000.016000   57000.000711      0.000000   93000.005855   60000.004800   42000.003048
#3  143000.000171   36000.008000  93000.005855       0.000000  153000.000265  135000.001070
#4   10000.012800  117000.000962  60000.004800  153000.000265       0.000000   18000.001778
#5    8000.036000   99000.000247  42000.003048  135000.001070   18000.001778       0.000000

dist_z = pd.DataFrame(euclidean_distances(z_norm, z_norm))
print("Euclidian distances z-score:\n", dist_z)
#Euclidian distances z-score:
#          0         1         2         3         4         5
#0  0.000000  2.679906  2.786918  2.322172  1.080185  1.607658
#1  2.679906  0.000000  1.087767  1.701847  2.113493  1.642660
#2  2.786918  1.087767  0.000000  2.654089  1.865272  1.260089
#3  2.322172  1.701847  2.654089  0.000000  2.506812  2.428976
#4  1.080185  2.113493  1.865272  2.506812  0.000000  0.606096
#5  1.607658  1.642660  1.260089  2.428976  0.606096  0.000000

dist_min_max = pd.DataFrame(euclidean_distances(min_max_norm, min_max_norm))
print("Euclidian distances min-max:\n", dist_min_max)
#Euclidian distances mix-max:
#           0         1         2         3         4         5
#0  0.000000  1.043892  1.052044  0.950883  0.405305  0.602274
#1  1.043892  0.000000  0.435222  0.644487  0.851704  0.670306
#2  1.052044  0.435222  0.000000  1.024743  0.716789  0.485135
#3  0.950883  0.644487  1.024743  0.000000  1.025000  0.979373
#4  0.405305  0.851704  0.716789  1.025000  0.000000  0.232036
#5  0.602274  0.670306  0.485135  0.979373  0.232036  0.000000
