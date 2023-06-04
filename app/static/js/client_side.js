$(document).ready(function(){
  
  let slideIndex = 0;
  showSlides();
  
  function showSlides() {
    let i;
    let slides = document.getElementsByClassName("mySlides");
    for (i = 0; i < slides.length; i++) {
      slides[i].style.display = "none";
    }
    slideIndex++;
    if (slideIndex > slides.length) {slideIndex = 1}
    slides[slideIndex-1].style.display = "block";
    setTimeout(showSlides, 3000); // Change image every 2 seconds
  }

  // -[Animasi Scroll]---------------------------
  
  $(".navbar a, footer a[href='#halamanku']").on('click', function(event) {
    if (this.hash !== "") {
      event.preventDefault();
      var hash = this.hash;
      $('html, body').animate({
        scrollTop: $(hash).offset().top
      }, 900, function(){
        window.location.hash = hash;
      });
    } 
  });
  
  $(window).scroll(function() {
    $(".slideanim").each(function(){
      var pos = $(this).offset().top;
      var winTop = $(window).scrollTop();
        if (pos < winTop + 600) {
          $(this).addClass("slide");
        }
    });
  });

  $(function () {
    $(document).scroll(function () {
      var $nav = $(".navbar-fixed-top");
      $nav.toggleClass('scrolled', $(this).scrollTop() > $nav.height());
    });
  });

  
  // -[Prediksi Model]---------------------------
  
  // Fungsi untuk memanggil API ketika tombol prediksi ditekan
  $("#prediksi_submit").click(function(e) {
    e.preventDefault();
	
	// Set data pengukuran bunga iris dari input pengguna
    var input_luas_bangunan = $("#range_luas_bangunan").val(); 
	var input_luas_tanah  = $("#range_luas_tanah").val(); 
	var input_kamar_tidur = $("#range_kamar_tidur").val(); 
	var input_kamar_mandi  = $("#range_kamar_mandi").val();
  var input_garasi  = $("#range_garasi").val();  

	// Panggil API dengan timeout 1 detik (1000 ms)
    setTimeout(function() {
	  try {
			$.ajax({
			  url  : "/api/deteksi",
			  type : "POST",
			  data : {"luas_bangunan" : input_luas_bangunan,
					  "luas_tanah"  : input_luas_tanah,
					  "kamar_tidur" : input_kamar_tidur,
					  "kamar_mandi"  : input_kamar_mandi,
            "garasi"  : input_garasi,
			         },
			  success:function(res){
				// Ambil hasil prediksi spesies dan path gambar spesies dari API
				res_data_prediksi   = res['prediksi']
				res_gambar_prediksi = res['gambar_prediksi']
				
				// Tampilkan hasil prediksi ke halaman web
			    generate_prediksi(res_data_prediksi,res_gambar_prediksi); 
			  }
			});
		}
		catch(e) {
			// Jika gagal memanggil API, tampilkan error di console
			console.log("Gagal !");
			console.log(e);
		} 
    }, 1000)
    
  })

  function rubah(angka){
    var reverse = angka.toString().split('').reverse().join(''),
    ribuan = reverse.match(/\d{1,3}/g);
    ribuan = ribuan.join('.').split('').reverse().join('');
    return ribuan;
  }

  // Fungsi untuk menampilkan hasil prediksi model
  function generate_prediksi(data_prediksi, image_prediksi) {
    var str="";
    str += "<h3>Hasil Prediksi </h3>";
    str += "<br>";
    str += "<img src='" + image_prediksi + "' width=\"450\" height=\"250\"></img>"
    str += "<br>";
    str += "<h4> Prediksi harga Rumah Kamu Adalah :</h4>";
    str += "<h3>"+ "± " +"RP. "+ rubah(data_prediksi) + "</h3>";
    $("#hasil_prediksi").html(str);
  }  
  
})
  
