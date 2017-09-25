from sklearn.cluster import AgglomerativeClustering
import numpy as np
from utility import construct_doc_matrix


class Evaluator():
    @staticmethod
    def compute_f1(dataset, bpr_optimizer):
        """
        perform Hierarchy Clustering on doc embedding matrix
        for name disambiguation
        use cluster-level mean F1 for evaluation
        """
        D_matrix = construct_doc_matrix(bpr_optimizer.paper_latent_matrix,
                                        dataset.paper_list)
        true_cluster_size = len(set(dataset.label_list))
        y_pred = AgglomerativeClustering(n_clusters = true_cluster_size,
                                         linkage = "average",
                                         affinity = "cosine").fit_predict(D_matrix)

        TP = 0.0  # Pairs Correctly Predicted To SameAuthor
        TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
        TP_FN = 0.0  # Total Pairs To SameAuthor

        for i in range(D_matrix.shape[0]):
            for j in range(i + 1, D_matrix.shape[0]):
                if dataset.label_list[i] == dataset.label_list[j]:
                    TP_FN += 1
                if y_pred[i] == y_pred[j]:
                    TP_FP += 1
                if (dataset.label_list[i] == dataset.label_list[j]) \
                        and (y_pred[i] == y_pred[j]):
                    TP += 1
        if TP == 0:
            pairwise_precision = 0
            pairwise_recall = 0
            pairwise_f1 = 0
        else:
            pairwise_precision = TP / TP_FP
            pairwise_recall = TP / TP_FN
            pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)


        # true_label_dict = {}
        # for idx, true_lbl in enumerate(dataset.label_list):
        #     if true_lbl not in true_label_dict:
        #         true_label_dict[true_lbl] = [idx]
        #     else:
        #         true_label_dict[true_lbl].append(idx)
        #
        # predict_label_dict = {}
        # for idx, pred_lbl in enumerate(y_pred):
        #     if pred_lbl not in predict_label_dict:
        #         predict_label_dict[pred_lbl] = [idx]
        #     else:
        #         predict_label_dict[pred_lbl].append(idx)
        #
        # # compute cluster-level F1
        # # let's denote C(r) as clustering result and T(k) as partition (ground-truth)
        # # construct r * k contingency table
        # r_k_table = []
        # for v1 in predict_label_dict.itervalues():
        #     k_list = []
        #     for v2 in true_label_dict.itervalues():
        #         N_ij = len(set(v1).intersection(v2))
        #         k_list.append(N_ij)
        #     r_k_table.append(k_list)
        # r_k_matrix = np.array(r_k_table)
        # r_num = int(r_k_matrix.shape[0])
        #
        # # compute F1 for each row C_i
        # sum_f1 = 0.0
        # for row in xrange(0, r_num):
        #     row_sum = np.sum(r_k_matrix[row,:])
        #     if row_sum != 0:
        #         max_col_index = np.argmax(r_k_matrix[row,:])
        #         row_max_value = r_k_matrix[row, max_col_index]
        #         prec = float(row_max_value) / row_sum
        #         col_sum = np.sum(r_k_matrix[:, max_col_index])
        #         rec = float(row_max_value) / col_sum
        #         row_f1 = float(2 * prec * rec) / (prec + rec)
        #         sum_f1 += row_f1
        #
        # average_f1 = float(sum_f1) / r_num
        # return average_f1

        return pairwise_precision, pairwise_recall, pairwise_f1
