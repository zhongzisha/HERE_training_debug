
import sys,os,glob,shutil
import idr_torch

from collections import defaultdict

class BKTree:
    def __init__(self, dist_func):
        self.dist_func = dist_func
        self.tree = {}

    def insert(self, word):
        node = self.tree
        while True:
            if not node:
                node.update({0: word})
                break
            dist = self.dist_func(word, node[0])
            node = node.setdefault(dist, {})

    def search(self, word, max_dist):
        candidates = [self.tree]
        result = []
        while candidates:
            node = candidates.pop()
            dist = self.dist_func(word, node[0])
            if dist <= max_dist:
                result.append(node[0])
            candidates.extend(child for d, child in node.items() if d > 0 and abs(d - dist) <= max_dist)
        return result

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def longestCommonSubsequence(text1: str, text2: str) -> int:
        # Get the lengths of both input strings
        len_text1, len_text2 = len(text1), len(text2)
      
        # Initialize a 2D array (list of lists) with zeros for dynamic programming
        # The array has (len_text1 + 1) rows and (len_text2 + 1) columns
        dp_matrix = [[0] * (len_text2 + 1) for _ in range(len_text1 + 1)]
      
        # Loop through each character index of text1 and text2
        for i in range(1, len_text1 + 1):
            for j in range(1, len_text2 + 1):
                # If the characters match, take the diagonal value and add 1
                if text1[i - 1] == text2[j - 1]:
                    dp_matrix[i][j] = dp_matrix[i - 1][j - 1] + 1
                else:
                    # If the characters do not match, take the maximum of the value from the left and above
                    dp_matrix[i][j] = max(dp_matrix[i - 1][j], dp_matrix[i][j - 1])
      
        # The bottom-right value in the matrix contains the length of the longest common subsequence
        return dp_matrix[len_text1][len_text2]


if __name__ == '__main__':

    # Usage
    bk_tree = BKTree(levenshtein)
    svs_dir = '/data/zhongz2/KenData_256/svs'
    patches_dir = '/data/zhongz2/KenData_256/patches'
    files = glob.glob(os.path.join(patches_dir, '*.h5'))
    all_prefixes = sorted([os.path.basename(f).replace('.h5', '') for f in files])
    for word in all_prefixes:
        bk_tree.insert(word)

    similar_cases = {}
    similar_cases_all = set()
    for query in all_prefixes:
        matches = bk_tree.search(query, 1)  # Finds words within an edit distance of 1
        if len(matches) > 1:
            similar_cases[query] = matches

    removed_cases = []
    for k, v in similar_cases.items():
        if not os.path.exists(os.path.join(patches_dir, k+'.h5')):
            continue
        size0 = os.path.getsize(os.path.realpath(os.path.join(svs_dir, k + '.svs')))
        size1 = os.path.getsize(os.path.realpath(os.path.join(patches_dir, k+'.h5')))
        for vv in v:
            if k == vv:
                continue
            size0_1 = os.path.getsize(os.path.realpath(os.path.join(svs_dir, vv + '.svs')))
            size1_1 = os.path.getsize(os.path.realpath(os.path.join(patches_dir, vv+'.h5')))
            if size0 == size0_1 and size1 == size1_1:
                print(k, vv)
                os.system('rm "{}/{}.h5"'.format(patches_dir, vv))
                removed_cases.append((k, vv))

