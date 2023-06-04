# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from joblib import load

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Nilai default untuk variabel input atau features (X) ke model
	input_luas_bangunan = 0
	input_luas_tanah  = 0
	input_kamar_tidur = 0
	input_kamar_mandi  = 0
	input_garasi  = 0
	
	if request.method=='POST':
		# Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
		input_luas_bangunan = float(request.form['luas_bangunan'])
		input_luas_tanah  = float(request.form['luas_tanah'])
		input_kamar_tidur = int(request.form['kamar_tidur'])
		input_kamar_mandi  = int(request.form['kamar_mandi'])
		input_garasi = int (request.form['garasi'])
		
		# Prediksi kelas berdasarkan data pengukuran yg diberikan pengguna
		df_test = pd.DataFrame(data={
			"LB" : [input_luas_bangunan],
			"LT"  : [input_luas_tanah],
			"KT" : [input_kamar_tidur],
			"KM"  : [input_kamar_mandi],
			"GRS" : [input_garasi],
		})

		hasil_prediksi = model.predict(df_test[0:1])[0]

		# Set Path untuk gambar hasil prediksi
		if hasil_prediksi > 100000000 and hasil_prediksi< 1000000000: #Diantara 100 juta dan 1 Miliar
			gambar_prediksi = '/static/images/rumah_murah.jpeg'
		elif hasil_prediksi > 1000000000 and hasil_prediksi< 15000000000: #Diantara 1 Miliar dan 15 Miliar
			gambar_prediksi = '/static/images/rumah_sederhana.jpeg'
		elif hasil_prediksi > 15000000000: #Diatas 15 M
			gambar_prediksi = '/static/images/rumah_mahal.jpeg'
		
		# Return hasil prediksi dengan format JSON
		return jsonify({
			"prediksi": hasil_prediksi,
			"gambar_prediksi" : gambar_prediksi
		})

# =[Main]========================================

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	model = load('harga_rumah_knnregressor_fiks.model')

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)
	
	


