import csv
import itertools

from MiniMax.EvaluationFunctions.EvaluationFunctions import basic_ev_func, push_forward_ev_func, \
    push_to_opp_half_ev_func, group_prize_ev_func
from MiniMax.checkers_stud import ai_vs_ai


def report_data_generator():
    eval_functions = [basic_ev_func, push_forward_ev_func, push_to_opp_half_ev_func, group_prize_ev_func]
    depths = [1, 2, 3, 4, 5]

    # Badanie 1
    print("Badanie 1\n")
    with open("test_results/eval_func_1_results.csv", mode ="w") as file_1:
        iteration = 0
        writer = csv.writer(file_1)
        writer.writerow(["Czarny głębia", "Biały głębia", "Wynik"])
        for black_depth, white_depth in list(itertools.product(depths, depths)):
            print(f"Iteration ${iteration}")
            iteration += 1
            result = ai_vs_ai(black_depth, white_depth, eval_functions[0])
            writer.writerow([
                black_depth,
                white_depth,
                "Biały" if  result == -1 else "Czarny" if result == 1 else "Remis"
            ])

    #Badanie 2
    print("Badanie 2\n")
    with open("test_results/eval_func_2_results.csv", mode ="w") as file_1:
        iteration = 0
        writer = csv.writer(file_1)
        writer.writerow(["Czarny głębia", "Biały głębia", "Wynik"])
        for black_depth, white_depth in list(itertools.product(depths, depths)):
            print(f"Iteration ${iteration}")
            iteration += 1
            result = ai_vs_ai(black_depth, white_depth, eval_functions[1])
            writer.writerow([
                black_depth,
                white_depth,
                "Biały" if  result == -1 else "Czarny" if result == 1 else "Remis"
            ])

    #Badanie 3
    print("Badanie 3\n")
    with open("test_results/eval_func_3_results.csv", mode ="w") as file_1:
        iteration = 0
        writer = csv.writer(file_1)
        writer.writerow(["Czarny głębia", "Biały głębia", "Wynik"])
        for black_depth, white_depth in list(itertools.product(depths, depths)):
            print(f"Iteration ${iteration}")
            iteration += 1
            result = ai_vs_ai(black_depth, white_depth, eval_functions[2])
            writer.writerow([
                black_depth,
                white_depth,
                "Biały" if  result == -1 else "Czarny" if result == 1 else "Remis"
            ])

    #Badanie 4
    print("Badanie 4\n")
    with open("test_results/eval_func_4_results.csv", mode ="w") as file_1:
        iteration = 0
        writer = csv.writer(file_1)
        writer.writerow(["Czarny głębia", "Biały głębia", "Wynik"])
        for black_depth, white_depth in list(itertools.product(depths, depths)):
            print(f"Iteration ${iteration}")
            iteration += 1
            result = ai_vs_ai(black_depth, white_depth, eval_functions[3])
            writer.writerow([
                black_depth,
                white_depth,
                "Biały" if  result == -1 else "Czarny" if result == 1 else "Remis"
            ])

report_data_generator()