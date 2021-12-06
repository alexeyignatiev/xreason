#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## tree.py (reuses parts of the code of SHAP)
##
##  Created on: Dec 7, 2018
##      Author: Nina Narodytska
##      E-mail: narodytska@vmware.com
##

#
#==============================================================================
from anytree import Node, RenderTree,AsciiStyle
import json
import numpy as np
import xgboost as xgb
import math


#
#==============================================================================
class xgnode(Node):
    def __init__(self, id, parent = None):
        Node.__init__(self, id, parent)
        self.id = id  # The node value
        self.name = None
        self.left_node_id = -1   #  Left child
        self.right_node_id = -1  # Right child
        self.missing_node_id = -1

        self.feature = -1
        self.threshold = -1

        self.cover = -1
        self.values = -1

    def __str__(self):
        pref = ' ' * self.depth
        if (len(self.children) == 0):
            return (pref+ "leaf: {}  {}".format(self.id, self.values))
        else:
            if(self.name is None):
                return (pref+ "{} f{}<{}".format(self.id, self.feature, self.threshold))
            else:
                return (pref+ "{} \"{}\"<{}".format(self.id, self.name, self.threshold))


#
#==============================================================================
def build_tree(json_tree, node = None, feature_names = None, inverse = False):
    def max_id(node):
        if "children" in node:
            return max(node["nodeid"], *[max_id(n) for n in node["children"]])
        else:
            return node["nodeid"]
    m = max_id(json_tree) + 1
    def extract_data(json_node, root = None, feature_names = None):
        i = json_node["nodeid"]
        if (root is None):
            node = xgnode(i)
        else:
            node = xgnode(i, parent = root)
        node.cover = json_node["cover"]
        if "children" in json_node:

            node.left_node_id = json_node["yes"]
            node.right_node_id = json_node["no"]
            node.missing_node_id = json_node["missing"]
            node.feature = json_node["split"]
            if (feature_names is not None):
                node.name = feature_names[node.feature]
            node.threshold = json_node["split_condition"]
            for c, n in enumerate(json_node["children"]):
                child = extract_data(n, node, feature_names)
        elif "leaf" in json_node:
            node.values  = json_node["leaf"]
            if(inverse):
                node.values = -node.values
        return node

    root = extract_data(json_tree, None, feature_names)
    return root


#
#==============================================================================
def walk_tree(node):
    if (len(node.children) == 0):
        # leaf
        print(node)
    else:
        print(node)
        walk_tree(node.children[0])
        walk_tree(node.children[1])


#
#==============================================================================
def scores_tree(node, sample):
    if (len(node.children) == 0):
        # leaf
        return node.values
    else:
        feature_branch = node.feature
        sample_value = sample[feature_branch]
        assert(sample_value is not None)
        if(sample_value < node.threshold):
            return scores_tree(node.children[0], sample)
        else:
            return scores_tree(node.children[1], sample)


#
#==============================================================================
class TreeEnsemble:
    """ An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """
    def __init__(self, model, feature_names = None, nb_classes = 0):
        self.model_type = "xgboost"
        self.original_model = model.get_booster()
        self.base_offset = None
        json_trees = get_xgboost_json(self.original_model)
        self.trees = [build_tree(json.loads(t), None, feature_names) for t in json_trees]
        if(nb_classes == 2):
            # NASTY trick for binary
            # We change signs of values in leaves so that we can just sum all the values in leaves for class X
            # and take max to get the right class
            self.otrees = [build_tree(json.loads(t), None, feature_names, inverse = True) for t in json_trees]
            self.itrees = [build_tree(json.loads(t), None, feature_names) for t in json_trees]
            self.trees = []
            for i,_ in enumerate(self.otrees):
                self.trees.append(self.otrees[i])
                self.trees.append(self.itrees[i])
        self.feature_names = feature_names
    def print_tree(self):
        for i,t in enumerate(self.trees):
            print("tree number: ", i)
            walk_tree(t)

    def invert_tree_prob(self, node):
        if (len(node.children) == 0):
            node.values = -node.values
            return node
        else:
            self.invert_tree_prob(node.children[0])
            self.invert_tree_prob(node.children[1])
        return node
    def predict(self, samples, nb_classes):
        # https://github.com/dmlc/xgboost/issues/1746#issuecomment-290130695
        prob = []
        for sample in np.asarray(samples):
            scores = []
            for i,t in enumerate(self.trees):
                s = scores_tree(t, sample)
                scores.append((s))
            scores = np.asarray(scores)
            class_scores = []
            if (nb_classes == 2):

                for i in range(nb_classes):
                    class_scores.append(math.exp(-(scores[i::nb_classes]).sum())) # swap signs back as we had to use this trick in the contractor
                s0 =  class_scores[0]
                s1 =  class_scores[1]
                v0 =  1/(1 + s0)
                v1 =  1/(1 + s1)
                class_scores[0] = v0
                class_scores[1] = v1
            else:
                for i in range(nb_classes):
                    class_scores.append(math.exp((scores[i::nb_classes]).sum()))
            class_scores = np.asarray(class_scores)
            prob.append(class_scores/class_scores.sum())
        return np.asarray(prob).reshape((-1, nb_classes))


#
#==============================================================================
def get_xgboost_json(model):
    """ REUSED FROM SHAP
        This gets a JSON dump of an XGBoost model while ensuring the feature names are their indexes.
    """
    fnames = model.feature_names
    model.feature_names = None
    json_trees = model.get_dump(with_stats=True, dump_format="json")
    model.feature_names = fnames
    return json_trees
