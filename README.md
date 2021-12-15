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

## GEREKSİNİMLER
  '''
  - GetOldTweets3==0.0.11
  - keras==2.7.0
  - keras_nightly==2.5.0.dev2021032900
  - matplotlib==3.1.3
  - nltk==3.4.5
  - numpy==1.20.3
  - pandas==1.3.3
  - scikit_learn==1.0.1
  - snscrape==0.3.4
  - textblob==0.15.3
  - wordcloud==1.6.0
  '''

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

## YAPAY SİNİR AĞI (ANN) MODELLERİ
Yapay sinir ağlarında sadece batch size değeri sabit tutulmuştur. Batch size değerine ‘1’ verilmiş ve her modelde bu değer kullanılmıştır. Learning rate, momentum gibi değerlerde ise default değerler kullanıldı. Her model 1000 ve 5000 epoch çalıştırılmıştır. Yapay sinir ağı modellerinde 1 veya 2 gizli katmanlı modeller kullanılmıştır.
Gizli katmanlarda 4,8,16 veya 32 nöron kullanılmıştır. Eğer iki gizli katman varsa her iki katmanda da aynı nöron sayısı kullanılmıştır. 3 farklı kayıp fonksiyonu kullanıldı. Bunlar MSE, MAE ve Huber Loss kayıp fonksiyonlarıdır. 3 farklı optimizasyon fonksiyonu kullanıldı. Bunlar Adam, Rmsprop ve  Sigmoid optimizasyon fonksiyonlarıdır.
En düşük MAE değerine sahip yapay sinir ağı modeli, 1 gizli katmana sahip ve bu gizli katmanda da 4 nöronu vardı. 

En düşük MAE değerine sahip yapay sinir ağı modelinin sonuclari
![image](https://user-images.githubusercontent.com/46621453/146273563-d39847b3-051f-4833-99ee-431362740cc9.png)

En düşük MSE değerine sahip yapay sinir ağı modelinin sonuclari
![image](https://user-images.githubusercontent.com/46621453/146273586-f9fb2bd3-eef7-49f7-9042-dadbbc93b599.png)

## LSTM MODELLERİ
En düşük MSE değerine sahip LSTM modeli, 1 gizli katmana sahip ve bu gizli katmanlarda 8 nöronu vardı. Optimizasyon fonksiyonu Adam, kayıp fonksiyonu ise Huber loss ve veri seti ise rastgele olan veri setidir.

En düşük MAE değerine sahip yapay sinir ağı modelinin sonuclari
![image](https://user-images.githubusercontent.com/46621453/146273697-04d3d166-91fa-4561-9085-cb3f31db2223.png)

En düşük MSE değerine sahip yapay sinir ağı modelinin sonuclari
![image](https://user-images.githubusercontent.com/46621453/146273712-02b8badb-cc6e-4cad-847f-b4408e293ceb.png)

# SONUÇLARIN KARŞILAŞTIRILMASI
![image](https://user-images.githubusercontent.com/46621453/146274685-63ab5e90-3643-4008-9c60-6ca8cc8b06c8.png)
![image](https://user-images.githubusercontent.com/46621453/146274691-7bbbfba0-f0d8-4d24-9cca-dea7c6cc77ec.png)





















