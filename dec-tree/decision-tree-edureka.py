import pandas

training_data = [
    ["Green", 3, "Mango"],
    ["Yellow", 3, "Mango"],
    ["Red", 1, "Grape"],
    ["Red", 1, "Grape"],
    ["Yellow", 3, "Lemon"]
]

header = ["color", "diameter", "label"]


def unique_vals(rows, col):
    return set([row[col] for row in rows])


def unique_vals_df(dataset, col):
    return set(dataset.iloc[:, col])


############
# Demo:
# unique_vals(training_data,0)
# unique_vals(training_data,1)
############


def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def class_counts_df(dataset):
    counts = {}
    for entry in dataset.iloc[:, -1]:
        if entry not in counts:
            counts[entry] = 0
        counts[entry] += 1
    return counts


############
# Demo:
# class_counts(training_data)
############


def is_numeric(val):
    return isinstance(val, int) or isinstance(val, float)


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s ?" % (header[self.column], condition, str(self.value))


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        impurity -= prob_of_label ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertatinty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns
    for col in range(n_features):  # for each feature
        values = set([row[col] for row in rows])  # unique value in the column
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)  # split
        if len(true_rows) == 0 or len(false_rows) == 0:
            continue  # skip

        gain = info_gain(true_rows, false_rows, current_uncertatinty)

        if gain >= best_gain:
            best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def built_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)
    true_branch = built_tree(true_rows)
    false_branch = built_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing + str(node.question))

    print(spacing + "--> True:")
    print_tree(node.true_branch, spacing + "   ")

    print(spacing + "--> False:")
    print_tree(node.false_branch, spacing + "   ")


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = float(sum(counts.values()))
    probs = {}
    for label in counts.keys():
        probs[label] = str(int(counts[label] / total * 100)) + "%"
    return probs


if __name__ == "__main__":
    # my_tree = built_tree(training_data)
    # print_tree(my_tree)
    #
    # # evaluate
    # testing_data = [
    #     ["Green", 3, "Mango"],
    #     ["Yellow", 4, "Apple"],
    #     ["Red", 2, "Grape"],
    #     ["Red", 1, "Grape"],
    #     ["Yellow", 3, "Lemon"],
    # ]
    # for row in testing_data:
    #     print("Actual: %s. Predicted: %s" % (row[-1], print_leaf(classify(row, my_tree))))
    #################################
    dataset = pandas.read_csv("weather.csv")
    # print(dataset.columns)
    # print(len(training_data[0]))
    # for i in range(len(dataset.columns)):
    #     print(unique_vals_df(dataset, i))
    # for i in range(len(training_data[0])):
    #     print(unique_vals(training_data, i))
    print(class_counts(training_data))
    print(class_counts_df(dataset))
###TODO
# add support for missing values
# prune the tree to prevent overfitting
#  add support for regression
