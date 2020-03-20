# Pengenalan Pola Huruf dengan Perceptron & Backpropagation

Program ini melakukan training dan testing pola huruf dengan menggunakan algoritma **perceptron** dan **backpropagation**. Data yang digunakan berdasarkan buku dari Laurene Fausett, "Fundamentals Of Neural Networks" pp.71-76 character recognition example 2.14. Referensi: https://github.com/ismaildurmaz/perceptron-letter-detection

## Requirements:
Python 3, Tkinter

## Run:
python perceptron.py
python backpropagation.py

## User Interface:

![Alt text](https://ibb.co/R0Lz5mf)

**Load Button**: Training file is loading into view<br/>
**Save Button**: Saving file is saving to file. (**Attention: Files first character must be target letter**) <br/>
**Clear Button**: Clears the view<br/>
**Learning Rate**: Perceptron learnign rate value. It must be 0 < LR < 1<br/>
**Threshold**: Perceptron threshold value<br/>
**Iterasi Maksimum**: Perceptron maximum iterations value<br/>
**Train Button**: Training folder (data folder)<br/>

**Grid**: Mouse click swaps on/off neurons<br/>
**Test Button**: Testing current view<br/>
**Result**: Current testing view result<br/>

