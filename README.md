# Exchange-Rate-Forecasting-Project-With-Python
# ÖZET
Bu çalışmada çeşitli yapay zekâ yöntemleriyle bir sonraki güne ait döviz kuru tahmin etme çalışması yapılmıştır. Kullanılan yöntemler çoklu regresyon (MLR), yapay sinir ağları (ANN) ve LSTM ağlarıydı. Kullanılan veri setinde ise altı farklı parametre vardı. Bu parametreler ise 
- 2013-2021 arası günlük petrol fiyatları
- 2013-2021 arası günlük altın fiyatları
- Eski döviz kuru değerleri
- Twitter üzerinden günlük **'turkish economy'** içeren paylaşımlara yapılan duygu analizi
- Aylık Merkez Bankası yüzdesel faizi
- Aylık Tüketici fiyat endeksi yüzdesi

Bir güne ait yukarıdaki bu parametrelerin değerleri verildi ve bir sonraki günün döviz kuru değeri tahmin edilmeye çalışıldı.Üç farklı yöntem kullanıldı, sonuçlar elde edinildi ve performansları karşılaştırıldı.

# VERİ TOPLAMA
- Kullanılan her parametrenin için 1 Ocak 2013 tarihinden 1 Nisan 2021 tarihine kadar olan bütün değerleri toplanmıştır.
- Döviz kuru olarak USD/TRY dövizi kullanıldı. 
- Altın fiyatlarında ise gram altın(GAU) değerleri toplandı.
- Petrol fiyatlarında, ham petrol fiyatları kullanıldı.
- TÜFE değişimi ve Faiz değerlerinde ise aylık yüzdesel değerleri kullanıldı. 
- Duygu analizinde Twitter üzerinden ‘turkish economy’ araması ile çıkan 2013 yılından itibaren bütün tweetler toplanmıştır. Her tweetin duygu analizi yapılmış ve her tweetin ne kadar pozitif veya ne kadar negatif ya da nötr mü olduğu sayısal olarak ifade edilmiştir.

# VERİ ÖNİŞLEME
NULL veriler için öncelikle hangi parametrede ise o parametrenin ortalaması ve sonra da standart sapması verilmişti. Ancak veri seti uzun yıllara dayanan bir veri seti olduğundan ve Türkiye ekonomisi gelişmekte olan bir ekonomi olduğundan büyük dalgalanmalar ve minimum ve maksimum değerler arası çok fazla fark vardı. Bu yüzden NULL veriler için ortalama ve standart sapma kullanmak performansı düşürüyordu. Bu yüzden NULL veriler için bir önceki günün değerleri kullanıldı.Normalizasyon işlemi için de Min - Max normalizasyon kullanılmıştır. Eğitim aşamasında veriler küçükten büyüğe sıralı, büyükten küçüğe sıralı ve rastgele olmak üzere üç farklı şekilde verilmiştir.

# KULLANILAN MODELLER VE SONUÇLAR
## ÇOKLU REGRESYON MODELLERİ (MULTIPLE LINEAR REGRESSION)
3 farklı çoklu regresyon modeli kullanılmıştır. Verilerin sıralanış şekline göre üç farklı veri seti oluşturulmuştu(GAU) 
![image](https://user-images.githubusercontent.com/46621453/146272689-4df65282-3e68-48ad-8f81-3935bf2c120b.png)











