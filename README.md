# Crosswalk detection project 

Algorytm składa się z 2 głównych części: klasyfikacji oraz detekcji.

## Klasyfikacja
Opierałem się na szkielecie z zajęć. Wczytanie i sparsowanie danych wejściowych w postaci pliku xml wykonałem w funkcji load_data().  
  
  
Znaki oznaczone jako 'crosswalk' zajmujące min. 10% szerokości i 10% wysokości zdjęcia przypisywane są do klasy 1, pozostałe do klasy 0.  
  
  Tylko 10% zdjęć z bazy danych zawierało przejścia dla pieszych, więc 20% całej bazy było moim zbiorem testowym, natomiast zbiór uczący stanowiła część pozostałej bazy, a nie całe pozostałe 80%, gdyż dla takiego zboru uczącego dobrze klasyfikowane było tylko 1 z 18 testowych przejść dla pieszych.

Uzyskana dokładność oraz macierz pomyłek:
<p align="center"> <img src="https://i.imgur.com/YljNuef.png" /> </p>

A więc 9 z 18 znaków przejść dla pieszych ze zbioru testowego zostało prawidłowo przyporządkowane.  
  
  Nie trzeba wykonywać uczenia, gdyż jego efekty są już zapisane w pliku voc.npy


## Detekcja

Detekcji dokonuję poprzez detekcję kształtów funkcjami biblioteki OpenCV. 

<div align="justify">Jeśli dane zdjęcie zostało sklasyfikowane jako przejdzie dla pieszych ('label' == 1) to konwertowane jest do skali szarości, a następnie progowane. Wykorzystałem 2 wartości progowe, gdyż jedna jest dla małych znaków, a druga dla dużych. Następnie korzystając z funkcji findContours() zwracamy kontury odnalezionych kształtów, a następnie za pomocą aproksymacji wielomianu otrzymujemy liczbę punktów charakterystycznych danego kształtu. Dla małych znaków ich liczba wynosi 4, jednak perspektywa zdjęcia zwłaszcza dla dużych znaków sprawia, że aby wykrywać większość znaków, tych punktów charakterystycznych może też być 5 albo 7. To rodzi niestety problem, że czasami jakieś inne obiekty też mogą zostać wykryte, stąd warunki na powierzchnię kształtu (contourArea) oraz stosunek szerokości do wysokości boundboxa opisującego kształt. Jeśli spełnione zostaną te warunki to rysowany jest kontur odnalezionego kształtu na zdjęciu oraz printowane współrzędne.  
</div>  <br />

## Podsumowanie
 
<div align="justify"> Z 18 przejść dla pieszych ze zbioru testowego jako przejście dla pieszych zostało sklasyfikowane 9 (drugie 9 sklasyfikowane jako nie przejście dla pieszych). Natomiast detekcja przebiegła poprawnie dla 6 z nich, jeden znak został źle zlokalizowany, dwa nie wykryte na zdjęciu oraz niepoprawna detekcja jednego zdjęcia błędnie przypisanego do klasy przejść dla pieszych.</div> 
 <br /> <p align="center"> <img src="https://i.imgur.com/N0kaaJu.png" /> </p>

 
