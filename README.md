# Exchange-Rate-Forecasting-Project-With-Python
# ÖZET
Bu çalışmada çeşitli yapay zekâ yöntemleriyle bir sonraki güne ait döviz kuru tahmin etme çalışması yapılmıştır. Kullanılan yöntemler çoklu regresyon(MLR), yapay sinir ağları(ANN) ve LSTM ağlarıydı. Kullanılan veri setinde ise altı farklı parametre vardı. Bu parametreler ise 
- 2013-2021 arası günlük petrol fiyatları
- 2013-2021 arası günlük altın fiyatları
- Eski döviz kuru değerleri
- Twitter üzerinden günlük **'turkish economy'** içeren paylaşımlara yapılan duygu analizi
- Aylık Merkez Bankası yüzdesel faizi
- Aylık Tüketici fiyat endeksi yüzdesi

Bir güne ait yukarıdaki bu parametrelerin değerleri verildi ve bir sonraki günün döviz kuru değeri tahmin edilmeye çalışıldı.Üç farklı yöntem kullanıldı, sonuçlar elde edinildi ve performansları karşılaştırıldı.

