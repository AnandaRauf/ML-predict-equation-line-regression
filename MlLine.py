import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression 
print("-----------------------------------------------------\n")
print("Machine Learning Equation Line Regression\n")
devby= "Ananda Rauf Maududi"
devdate = "20 Desember 2020"
version = "1.0"
print("Developed by:",devby)
print("Developed date:",devdate)
print("Version:",version)
print("-----------------------------------------------------\n")


# Persamaan garis regresi dari data tinggi badan dan data berat badan(kg)

data_tinggi_badan = [151,174,138,186,128,136,179,163,152,131]
urutdata1 = sorted(data_tinggi_badan)
print("Data tinggi badan Mahasiswa/i yang sudah terurut:",urutdata1)
data_berat_badan = [63,81,56,91,47,57,76,72,62,48]
urutdata2 = sorted(data_berat_badan)
print("Data berat badan Mahasiswa/i yang sudah terurut:",urutdata2)

# Proses buat grafik persamaan Linear Regresi

# data tinggi badan paling depan dengan berat badan paling depan yang belum terurut untuk cara menghitung dan menentukan garis.

data = pd.DataFrame([[151,63],[174,81],[138,56],[186,91],[128,47],[136,57],[179,76],[163,72],[152,62],[131,48]])

# Data tinggi badan dan berat badan kita ganti menjadi variabel x (Tinggi badan) dan y (Berat badan)
data.columns = ['x','y']

x_data = data['x'].values[:,np.newaxis]
y_data = data['y'].values

# Buat Garis linear

linearmodels = LinearRegression()

linearmodels.fit(x_data,y_data) 
print("Coefficient :"+str(linearmodels.coef_))
print("Intercept :"+str(linearmodels.intercept_))

# Prediksi garis linear 

test_x = [[170],[171]] # < Data yang akan di prediksi

# Hasil prediksi 

prediksi_data = linearmodels.predict(test_x)
print(prediksi_data) # < hasil prediksi data

# Untuk menampilkan hasil prediksi garis linear prediksi dan tabel linear regresi 

line_regresi = linearmodels.predict(x_data)
hasil_line = pd.DataFrame({'x': data['x'],'y':line_regresi})
plt.scatter(data['x'],data['y'])
plt.plot(data['x'],data['y'],color='red',linewidth=1) #color red untuk memberikan warna garis linear regresi
plt.xlabel("Tinggi badan dalam cm")
plt.ylabel("Berat badan dalam kg")
plt.show()




