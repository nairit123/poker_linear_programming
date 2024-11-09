#!/usr/bin/env python3

import os
import argparse
import json
import gurobipy as gurobi
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

def solve_problem_2_1(game):
    for player in [1, 2]:
        # Create a new model for each player
        m = gurobi.Model("game_value_p1_" + str(player))
        # sequence is tuple of id corresponding to decision point and action
        # sequence -> index
        seqtoind1 = dict()
        seqtoind1[("null", "null")] = 0
        i = 1
        constraints_1 = 1
        for entry in game["decision_problem_pl1"]:
            constraints_1 += 1
            if entry["type"] == "decision":
                for action in entry["actions"]:
                    seqtoind1[(entry["id"], action)] = i
                    i += 1
        seqtoind2 = dict()
        seqtoind2[("null", "null")] = 0
        j = 1
        constraints_2 = 1
        for entry in game["decision_problem_pl2"]:
            constraints_2 += 1
            if entry["type"] == "decision":
                for action in entry["actions"]:
                    seqtoind2[(entry["id"], action)] = j
                    j += 1

        Sigma_1 = i
        Sigma_2 = j

        A = np.zeros((i, j))

        for util in game["utility_pl1"]:
            row = seqtoind1[(util["sequence_pl1"][0], util["sequence_pl1"][1])]
            col = seqtoind2[(util["sequence_pl2"][0], util["sequence_pl2"][1])]
            value = util["value"]
            A[row][col] = value

        # implement constraints for each decision point of pl1 and the null sequence

        F1 = np.zeros((constraints_1, Sigma_1))
        f1 = np.zeros(constraints_1)

        # null sequence is 1
        F1[0][0] = 1
        f1[0] = 1
        i = 1
        for entry in game["decision_problem_pl1"]:
            # make actions 1 and parent -1
            if entry["type"] != "decision":
                continue
            if ("parent_sequence" in entry.keys() and entry["parent_sequence"] != None):
                parentind = seqtoind1[tuple(entry["parent_sequence"])]
            else:
                parentind = 0
            F1[i][parentind] = -1
            for action in entry["actions"]:
                actionind = seqtoind1[(entry["id"], action)]
                F1[i][actionind] = 1
            i += 1

        F2 = np.zeros((constraints_2, Sigma_2))
        f2 = np.zeros(constraints_2)

        # null sequence is 1
        F2[0][0] = 1
        f2[0] = 1
        i = 1
        for entry in game["decision_problem_pl2"]:
            # make actions 1 and parent -1
            if entry["type"] != "decision":
                continue
            if ("parent_sequence" in entry.keys() and entry["parent_sequence"] != None):
                parentind = seqtoind2[tuple(entry["parent_sequence"])]
            else:
                parentind = 0
            F2[i][parentind] = -1
            for action in entry["actions"]:
                actionind = seqtoind2[(entry["id"], action)]
                F2[i][actionind] = 1
            i += 1

        x = m.addMVar(shape=Sigma_1, lb=0, ub = GRB.INFINITY, name="x")  # x or y >= 0
        m.addConstr(x >= 0)
        y = m.addMVar(shape=Sigma_2, lb=0, ub = GRB.INFINITY, name="y")
        m.addConstr(y >= 0)

        if player == 1:
            v = m.addMVar(shape=constraints_2, lb = -GRB.INFINITY, ub = GRB.INFINITY, name="v")  # v free
            m.addConstr(A.T @ x - F2.T @ v >= 0, name = "P1_Constraint1")  # A^T * x - F2^T * v >= 0
            m.addConstr(F1 @ x == f1, name = "P1_Constraint2")  # F1 * x == f1

            m.setObjective(f2 @ v, GRB.MAXIMIZE)

        elif player == 2:
            v = m.addMVar(shape=constraints_1, lb = -GRB.INFINITY, ub = GRB.INFINITY, name="v")  # v free
            m.addConstr(-A @ y - F1.T @ v >= 0, "P2_Constraint1")  # -A * y - F1^T * v >= 0
            m.addConstr(F2 @ y == f2, "P2_Constraint2")  # F2 * y == f2

            m.setObjective(f1 @ v, GRB.MAXIMIZE)

        m.optimize()

def solve_problem_2_2(game):
    for player in [1, 2]:
        # Create a new model for each player
        m = gurobi.Model("game_value_p1_" + str(player))
        # sequence is tuple of id corresponding to decision point and action
        # sequence -> index
        seqtoind1 = dict()
        seqtoind1[("null", "null")] = 0
        i = 1
        constraints_1 = 1
        for entry in game["decision_problem_pl1"]:
            constraints_1 += 1
            if entry["type"] == "decision":
                for action in entry["actions"]:
                    seqtoind1[(entry["id"], action)] = i
                    i += 1
        seqtoind2 = dict()
        seqtoind2[("null", "null")] = 0
        j = 1
        constraints_2 = 1
        for entry in game["decision_problem_pl2"]:
            constraints_2 += 1
            if entry["type"] == "decision":
                for action in entry["actions"]:
                    seqtoind2[(entry["id"], action)] = j
                    j += 1

        Sigma_1 = i
        Sigma_2 = j

        A = np.zeros((i, j))

        for util in game["utility_pl1"]:
            row = seqtoind1[(util["sequence_pl1"][0], util["sequence_pl1"][1])]
            col = seqtoind2[(util["sequence_pl2"][0], util["sequence_pl2"][1])]
            value = util["value"]
            A[row][col] = value

        # implement constraints for each decision point of pl1 and the null sequence

        F1 = np.zeros((constraints_1, Sigma_1))
        f1 = np.zeros(constraints_1)

        # null sequence is 1
        F1[0][0] = 1
        f1[0] = 1
        i = 1
        for entry in game["decision_problem_pl1"]:
            # make actions 1 and parent -1
            if entry["type"] != "decision":
                continue
            if ("parent_sequence" in entry.keys() and entry["parent_sequence"] != None):
                parentind = seqtoind1[tuple(entry["parent_sequence"])]
            else:
                parentind = 0
            F1[i][parentind] = -1
            for action in entry["actions"]:
                actionind = seqtoind1[(entry["id"], action)]
                F1[i][actionind] = 1
            i += 1

        F2 = np.zeros((constraints_2, Sigma_2))
        f2 = np.zeros(constraints_2)

        # null sequence is 1
        F2[0][0] = 1
        f2[0] = 1
        i = 1
        for entry in game["decision_problem_pl2"]:
            # make actions 1 and parent -1
            if entry["type"] != "decision":
                continue
            if ("parent_sequence" in entry.keys() and entry["parent_sequence"] != None):
                parentind = seqtoind2[tuple(entry["parent_sequence"])]
            else:
                parentind = 0
            F2[i][parentind] = -1
            for action in entry["actions"]:
                actionind = seqtoind2[(entry["id"], action)]
                F2[i][actionind] = 1
            i += 1

        x = m.addMVar(shape=Sigma_1, vtype = GRB.BINARY, name="x")  # x or y >= 0
        y = m.addMVar(shape=Sigma_2, vtype = GRB.BINARY, name="y")  # x or y >= 0

        if player == 1:
            v = m.addMVar(shape=constraints_2, lb = -GRB.INFINITY, ub = GRB.INFINITY, name="v")  # v free
            m.addConstr(A.T @ x - F2.T @ v >= 0, name = "P1_Constraint1")  # A^T * x - F2^T * v >= 0
            m.addConstr(F1 @ x == f1, name = "P1_Constraint2")  # F1 * x == f1

            m.setObjective(f2 @ v, GRB.MAXIMIZE)

        elif player == 2:
            v = m.addMVar(shape=constraints_1, lb = -GRB.INFINITY, ub = GRB.INFINITY, name="v")  # v free
            m.addConstr(-A @ y - F1.T @ v >= 0, "P2_Constraint1")  # -A * y - F1^T * v >= 0
            m.addConstr(F2 @ y == f2, "P2_Constraint2")  # F2 * y == f2

            m.setObjective(f1 @ v, GRB.MAXIMIZE)

        m.optimize()

def solve_problem_2_3(game):
    for player in [1, 2]:
        # Create a new model for each player
        # sequence is tuple of id corresponding to decision point and action
        # sequence -> index
        seqtoind1 = dict()
        seqtoind1[("null", "null")] = 0
        i = 1
        constraints_1 = 1
        for entry in game["decision_problem_pl1"]:
            constraints_1 += 1
            if entry["type"] == "decision":
                for action in entry["actions"]:
                    seqtoind1[(entry["id"], action)] = i
                    i += 1
        seqtoind2 = dict()
        seqtoind2[("null", "null")] = 0
        j = 1
        constraints_2 = 1
        for entry in game["decision_problem_pl2"]:
            constraints_2 += 1
            if entry["type"] == "decision":
                for action in entry["actions"]:
                    seqtoind2[(entry["id"], action)] = j
                    j += 1

        Sigma_1 = i
        Sigma_2 = j
        # i is |Sigma_1|, j = |Sigma_2|

        A = np.zeros((i, j))

        for util in game["utility_pl1"]:
            row = seqtoind1[(util["sequence_pl1"][0], util["sequence_pl1"][1])]
            col = seqtoind2[(util["sequence_pl2"][0], util["sequence_pl2"][1])]
            value = util["value"]
            A[row][col] = value


        # implement constraints for each decision point of pl1 and the null sequence

        F1 = np.zeros((constraints_1, Sigma_1))
        f1 = np.zeros(constraints_1)

        # null sequence is 1
        F1[0][0] = 1
        f1[0] = 1
        i = 1
        for entry in game["decision_problem_pl1"]:
            # make actions 1 and parent -1
            if entry["type"] != "decision":
                continue
            if ("parent_sequence" in entry.keys() and entry["parent_sequence"] != None):
                parentind = seqtoind1[tuple(entry["parent_sequence"])]
            else:
                parentind = 0
            F1[i][parentind] = -1
            for action in entry["actions"]:
                actionind = seqtoind1[(entry["id"], action)]
                F1[i][actionind] = 1
            i += 1

        F2 = np.zeros((constraints_2, Sigma_2))
        f2 = np.zeros(constraints_2)

        # null sequence is 1
        F2[0][0] = 1
        f2[0] = 1
        i = 1
        for entry in game["decision_problem_pl2"]:
            # make actions 1 and parent -1
            if entry["type"] != "decision":
                continue
            if ("parent_sequence" in entry.keys() and entry["parent_sequence"] != None):
                parentind = seqtoind2[tuple(entry["parent_sequence"])]
            else:
                parentind = 0
            F2[i][parentind] = -1
            for action in entry["actions"]:
                actionind = seqtoind2[(entry["id"], action)]
                F2[i][actionind] = 1
            i += 1


        if player == 1:
            k1max = len([1 for entry in game["decision_problem_pl1"] if entry["type"] == "decision"])
            values = []

            for k1 in range(k1max + 1):
                m = gurobi.Model("game_value_p1_" + str(player))
                m.setParam('OutputFlag', 0)
                x = m.addMVar(shape=Sigma_1, lb = 0, name="x")  # x or y >= 0
                y = m.addMVar(shape=Sigma_2, lb = 0, name="y")  # x or y >= 0
                z = m.addMVar(shape = Sigma_1, vtype = GRB.BINARY)
                # constraint 3, 4
                for entry in game["decision_problem_pl1"]:
                    if entry["type"] == "decision":
                        if entry["parent_sequence"] == None:
                            for action in entry["actions"]:
                                ja = seqtoind1[(entry["id"], action)]
                                m.addConstr(x[ja] >= z[ja], name="3")
                        else:
                            pj = seqtoind1[tuple(entry["parent_sequence"])]
                            for action in entry["actions"]:
                                ja = seqtoind1[(entry["id"], action)]
                                m.addConstr(x[ja] >= x[pj] + z[ja] - 1, name="4")
                # constraint 5
                for entry in game["decision_problem_pl1"]:
                    if entry["type"] == "decision":
                        listofja = [seqtoind1[(entry["id"], action)] for action in entry["actions"]]
                        m.addConstr(sum([z[ja] for ja in listofja]) <= 1, name="5")
                # constraint 6
                listofja = []
                for entry in game["decision_problem_pl1"]:
                    if entry["type"] == "decision":
                        listofja += [seqtoind1[(entry["id"], action)] for action in entry["actions"]]
                m.addConstr(sum([z[ja] for ja in listofja]) >= k1, name="6")


                v = m.addMVar(shape=constraints_2, lb = -GRB.INFINITY, ub = GRB.INFINITY, name="v")  # v free
                m.addConstr(A.T @ x - F2.T @ v >= 0, name = "P1_Constraint1")  # A^T * x - F2^T * v >= 0
                m.addConstr(F1 @ x == f1, name = "P1_Constraint2")  # F1 * x == f1

                m.setObjective(f2 @ v, GRB.MAXIMIZE)
                m.setParam("MIPGap", 0.01)
                m.optimize()

                values.append(m.objVal)
                print("done with k1="+ str(k1) + " out of k1max = " + str(k1max))
            print(values)

            # Define x-coordinates as the index of the values
            x = list(range(len(values)))

            # Define y-coordinates as the values
            y = values

            # Create the scatter plot
            plt.scatter(x, y)

            # Add labels and title
            plt.xlabel('Index (X-coordinate)')
            plt.ylabel('Value (Y-coordinate)')
            plt.title('Scatter Plot of Index vs Value')

            # Display the plot
            plt.savefig("player_1_k_plot.png")

            plt.clf()

        elif player == 2:
            k2max = len([1 for entry in game["decision_problem_pl2"] if entry["type"] == "decision"])
            values = []

            for k2 in range(k2max + 1):

                m = gurobi.Model("game_value_p2_" + str(player))
                m.setParam('OutputFlag', 0)

                x = m.addMVar(shape=Sigma_1, lb = 0, name="x")  # x or y >= 0
                y = m.addMVar(shape=Sigma_2, lb = 0, name="y")  # x or y >= 0
                z = m.addMVar(shape = Sigma_2, vtype = GRB.BINARY)
                # constraint 3, 4
                for entry in game["decision_problem_pl2"]:
                    if entry["type"] == "decision":
                        if entry["parent_sequence"] == None:
                            for action in entry["actions"]:
                                ja = seqtoind2[(entry["id"], action)]
                                m.addConstr(y[ja] >= z[ja], name="3")
                        else:
                            pj = seqtoind2[tuple(entry["parent_sequence"])]
                            for action in entry["actions"]:
                                ja = seqtoind2[(entry["id"], action)]
                                m.addConstr(y[ja] >= y[pj] + z[ja] - 1, name="4")
                # constraint 5
                for entry in game["decision_problem_pl2"]:
                    if entry["type"] == "decision":
                        listofja = [seqtoind2[(entry["id"], action)] for action in entry["actions"]]
                        m.addConstr(sum([z[ja] for ja in listofja]) <= 1, name="5")
                # constraint 6
                listofja = []
                for entry in game["decision_problem_pl2"]:
                    if entry["type"] == "decision":
                        listofja += [seqtoind2[(entry["id"], action)] for action in entry["actions"]]
                m.addConstr(sum([z[ja] for ja in listofja]) >= k2, name="6")
                v = m.addMVar(shape=constraints_1, lb = -GRB.INFINITY, ub = GRB.INFINITY, name="v")  # v free
                m.addConstr(-A @ y - F1.T @ v >= 0, "P2_Constraint1")  # -A * y - F1^T * v >= 0
                m.addConstr(F2 @ y == f2, "P2_Constraint2")  # F2 * y == f2

                m.setObjective(f1 @ v, GRB.MAXIMIZE)
                m.setParam("MIPGap", 0.01)
                m.optimize()

                values.append(m.objVal)
                print("done with k2="+ str(k1) + " out of k2max = " + str(k1max))
            print(values)

            # Define x-coordinates as the index of the values
            x = list(range(len(values)))

            # Define y-coordinates as the values
            y = values

            # Create the scatter plot
            plt.scatter(x, y)

            # Add labels and title
            plt.xlabel('Index (X-coordinate)')
            plt.ylabel('Value (Y-coordinate)')
            plt.title('Scatter Plot of Index vs Value')

            # Display the plot
            plt.savefig("player_2_k_plot.png")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HW2 Problem 2 (Deterministic strategies)')
    parser.add_argument("--game", help="Path to game file", required=True)
    parser.add_argument(
        "--problem", choices=["2.1", "2.2", "2.3"], required=True)

    args = parser.parse_args()
    print("Reading game path %s..." % args.game)

    game = json.load(open(args.game))

    # Convert all sequences from lists to tuples
    for tfsdp in [game["decision_problem_pl1"], game["decision_problem_pl2"]]:
        for node in tfsdp:
            if isinstance(node["parent_edge"], list):
                node["parent_edge"] = tuple(node["parent_edge"])
            if "parent_sequence" in node and isinstance(node["parent_sequence"], list):
                node["parent_sequence"] = tuple(node["parent_sequence"])
    for entry in game["utility_pl1"]:
        assert isinstance(entry["sequence_pl1"], list)
        assert isinstance(entry["sequence_pl2"], list)
        entry["sequence_pl1"] = tuple(entry["sequence_pl1"])
        entry["sequence_pl2"] = tuple(entry["sequence_pl2"])

    print("... done. Running code for Problem", args.problem)

    if args.problem == "2.1":
        solve_problem_2_1(game)
    elif args.problem == "2.2":
        solve_problem_2_2(game)
    else:
        assert args.problem == "2.3"
        solve_problem_2_3(game)
