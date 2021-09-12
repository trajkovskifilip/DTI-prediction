import collections
import numpy as np
from os import walk


def load_datasets(dataset_type, only_selected=True):
    # read the interaction matrix for drugs and targets
    DT_matrix_file = dataset_type + '/initial_DT_adjacency_matrix.txt'
    DT_matrix = np.genfromtxt(DT_matrix_file, delimiter='\t', dtype=str)

    # get all unique drugs and targets names with order preserving, and all DTIs (those that have 1)
    all_drugs, all_targets, DTIs = get_drugs_targets_names(DT_matrix)

    drug_similarity_files = []
    target_similarity_files = []

    # read files of similarities
    if only_selected:
      print("Loading only selected similarity matrices")
      drug_similarity_selected = dataset_type + '/drug_similarities/selected_files.txt'
      drug_similarity_files = np.loadtxt(drug_similarity_selected, delimiter='\n', dtype=str, skiprows=0)

      target_similarity_selected = dataset_type + '/target_similarities/selected_files.txt'
      target_similarity_files = np.loadtxt(target_similarity_selected, delimiter='\n', dtype=str, skiprows=0)
    else:
      print("Loading all similarity matrices")
      all_files = next(walk(dataset_type + '/drug_similarities'), (None, None, []))[2]
      drug_similarity_files = [file for file in all_files if "selected_files" not in file]

      all_files = next(walk(dataset_type + '/target_similarities'), (None, None, []))[2]
      target_similarity_files = [file for file in all_files if "selected_files" not in file]

    # build the similarity matrices of multiple similarities
    drug_similarity_matrices = build_multiple_similarity_matrices(drug_similarity_files, dataset_type, 'D', len(all_drugs))
    target_similarity_matrices = build_multiple_similarity_matrices(target_similarity_files, dataset_type, 'T', len(all_targets))

    # Create DT_labeled tree (drug, target, label) with known and unknown interactions
    DT_labeled = tree()

    # read the DTIs (format of DTIs.txt is ex. hsa:10008	D00294) and label them with 1 (known interaction)
    with open(dataset_type + '/DTIs.txt', 'r') as f:
        for lines in f:
            line = lines.split()
            line[0] = line[0].replace(":", "")
            DT_labeled[line[1]][line[0]] = 1

    # build the tree with all possible pairs and assign labels
    labels = []
    DT_pairs = []
    for drug in all_drugs:
        for target in all_targets:
            pair = drug, target
            # add negative labels to non-interactive pairs
            if DT_labeled[drug][target] != 1:
                DT_labeled[drug][target] = 0
                label = 0
            else:
                label = 1

            labels.append(label)
            DT_pairs.append(pair)

    # X = all (drug, target) pairs, Y = labels
    X = np.asarray(DT_pairs)
    Y = np.asarray(labels)
    print('Dimensions of all pairs:', X.shape)

    return all_drugs, all_targets, drug_similarity_matrices, target_similarity_matrices, DTIs, DT_labeled, X, Y


def get_drugs_targets_names(DT):
    # remove the drugs and targets names from the matrix
    DTIs = np.zeros((DT.shape[0] - 1, DT.shape[1] - 1))

    drugs = []
    targets = []
    for i in range(1, DT.shape[0]):
        for j in range(1, DT.shape[1]):
            DTIs[i - 1][j - 1] = DT[i][j]
    
    for i in range(1, DT.shape[0]):
        targets.append(DT[i][0])
    
    for j in range(1, DT.shape[1]):
        drugs.append(DT[0][j])

    DTIs = np.array(DTIs, dtype=np.float64)

    print('Number of drugs:', len(drugs))
    print('Number of targets:', len(targets))

    return drugs, targets, DTIs


def build_multiple_similarity_matrices(files, dataset_type, D_or_T, length):
    similarity_matrices = []
    for i in range(0, len(files)):
        if D_or_T == 'D':
            similarity_matrix_file = dataset_type + '/drug_similarities/' + str(files[i])
        else:
            similarity_matrix_file = dataset_type + '/target_similarities/' + str(files[i])
        similarity_matrices.append(np.loadtxt(similarity_matrix_file, delimiter='\t', dtype=np.float64, skiprows=1, usecols=range(1, length + 1)))

    return similarity_matrices


def tree():
    return collections.defaultdict(tree)