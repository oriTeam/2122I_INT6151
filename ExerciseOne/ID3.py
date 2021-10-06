"""
Name: Pham Van Trong
Class: K28-KHMT
MSHV: 21025021

"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.is_leaf = False
        self.decision = ""


def compute_entropy(data, target):
    decisions_list = data[target].values.tolist()
    decisions = np.reshape(decisions_list, len(decisions_list)).tolist()
    vc = pd.Series(decisions).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc)).sum()


# Q2.1
def compute_information_gain(data, feature, target):
    feature_data = data[feature].tolist()
    indexes = np.unique(feature_data, return_index=True)[1]
    uniq_feature_list = [feature_data[index] for index in sorted(indexes)]
    # print("uniq_feature_list", uniq_feature_list)
    # print("uniq_feature_list", data[feature])
    info_gain = compute_entropy(data, target)
    for uniq_feature in uniq_feature_list:
        child_data = data[data[feature] == uniq_feature]
        child_entropy = compute_entropy(child_data, target)
        info_gain -= float(len(child_data)) / float(len(data)) * child_entropy
    return info_gain


# Build decision tree on X and y
# List of:
# node_index, node_feature[0..3], (feature_value -> child_index) : internal node
# leafnode: node_index, node_features = -1, Yes/No
def build_ID3(data, features, target):
    root = Node()
    max_info_gain = 0
    max_feature = None
    for feature in features:
        info_gain = compute_information_gain(data, feature, target)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_feature = feature
    root.value = max_feature
    # print ("\nMax feature attr",max_feat)

    feature_data = data[max_feature].tolist()
    indexes = np.unique(feature_data, return_index=True)[1]
    uniq_feature_list = [feature_data[index] for index in sorted(indexes)]
    # print("uniq_feature_list", uniq_feature_list)

    for uniq_feature in uniq_feature_list:
        child_data = data[data[max_feature] == uniq_feature]
        if compute_entropy(child_data, target) == 0:
            new_node = Node()
            new_node.is_leaf = True
            new_node.value = uniq_feature
            new_node.decision = np.unique(child_data[target[0]])[0]
            root.children.append(new_node)
        else:
            tmp_node = Node()
            tmp_node.value = uniq_feature
            new_features = features.copy()
            new_features.remove(max_feature)
            child = build_ID3(child_data, new_features, target)
            tmp_node.children.append(child)
            root.children.append(tmp_node)

    return root


def print_ID3_tree(root: Node, depth=0, index=0):
    for i in range(depth):
        print("---- ", end="")
    print(root.value, end="")
    if root.is_leaf:
        print(" -> ", root.decision, end="")
    print()
    for child in root.children:
        print_ID3_tree(child, depth + 1)


def predict(tree_root, X_test):
    samples = X_test.to_dict(orient='records')
    predictions = []
    for sample in samples:
        node = tree_root
        count = 0
        while not node.is_leaf:
            if count % 2 == 0:
                node = list(filter(lambda x: x.value == sample[node.value], node.children))[0]
            else:
                node = node.children[0]
            count = count + 1
        predictions.append(node.decision)

    return predictions


def main():
    df = pd.read_csv("./train.csv")
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    target = ['PlayTennis']

    # print(df[features])
    X = df[features].values
    y = df[target].values
    print("Input: ", X.shape, y.shape)
    print('===================================================================')

    # Q2.1
    print('Q2.1 - Information Gain:')
    print("Information Gain of Outlook: ", compute_information_gain(df, 'Outlook', target))
    print('===================================================================')

    # Q2.2
    print('Q2.2 - ID3 Tree:')
    root = build_ID3(df, features, target)
    print_ID3_tree(root)
    print('===================================================================')

    # Q2.3: Predict test
    print('Q2.3 - Predict test:')
    tf = pd.read_csv("./test.csv")
    X_test = tf[features]
    y_test = tf[target]
    y_pred = predict(root, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Y Predictions: ", y_pred)
    print("Y Test: ", np.reshape(y_test.values.tolist(), len(y_test.values.tolist())).tolist())
    print("Accuracy: ", accuracy)
    print('===================================================================')


if __name__ == "__main__":
    main()
