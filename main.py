from settings import *

if __name__ == "__main__":    
    print(" Ряд Фиббоначи(1) \n функция(2)")
    print(" Степенная функция(3) \n Послед:1,2,3,5,8,13(4) \n")
    command = input()
    if command == "1":
        sequence=[0, 1, 1, 2, 3, 5]
        file_name_w1 = "w1_fibonachi.bin"
        file_name_w2 = "w2_fibonachi.bin"
        p = 3
    if command == "2":
        sequence = [1, 0, -1, 0, 1, 0]
        file_name_w1 = "w1_period.bin"
        file_name_w2 = "w2_period.bin"
        p = 3
    if command == "3":
        sequence = [1, 4, 9, 16,25,36,49]
        file_name_w1 = "w1_step.bin" 
        file_name_w2 = "w2_step.bin"
        p = 4
    if command == "4":
        sequence = [1, 2, 3, 5, 8, 13]
        file_name_w1 = "w1_sequence_1.bin" 
        file_name_w2 = "w2_sequence_1.bin"
        p = 3
    print("Выерите : обучать(1) - предсказать(2)")
    lp_command = input()
    if lp_command == "1":
        w1, w2 = leraning(sequence ,p, error = 0.01, max_iter = 500000, m = 2, alpha =0.000015)
        save_w1(w1,file_name_w1)
        save_w2(w2,file_name_w2)
        predict(w1,w2, sequence,p,m=2,predict_n = 1)
    if lp_command == "2":
        w1 = read_matrix_w1(file_name_w1)
        w2 = read_matrix_w2(file_name_w2)
        print(w1)
        print(w2)
        predict(w1,w2, sequence,p,m=2,predict_n = 1)
    