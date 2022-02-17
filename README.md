# Crosswalk detection project

Algorytm składa się z 2 głównych części: klasyfikacji oraz detekcji.

## Klasyfikacja
Opierałem się na szkielecie z zajęć. Wczytanie i sparsowanie danych wejściowych w postaci pliku xml wykonałem w funkcji load_data().  
  
  
Znaki oznaczone jako 'crosswalk' zajmujące min. 10% szerokości i 10% wysokości zdjęcia przypisywane są do klasy 1, pozostałe do klasy 0.  

Uzyskana dokładność oraz macierz pomyłek:
<p align="center"> <img src="https://i.imgur.com/Q8WVAlG.png" /> </p>


## Detekcja
 
