import parser_helper
import embedding
import train_helper
import sampler
import eval_metric
import argparse
import os
import re


def get_file_list(dir, file_list):
    newDir = dir
    if os.path.isfile(dir):
        file_list.append(dir.decode('gbk'))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            get_file_list(newDir, file_list)
    return file_list


def main(file_path, latent_dimen, alpha, matrix_reg, num_epoch, sampler_method):
    """
    pipeline for representation learning for all papers for a given name reference
    """
    dataset = parser_helper.DataSet(file_path)
    paper_count = dataset.reader()

    bpr_optimizer = embedding.BprOptimizer(latent_dimen, alpha,
                                           matrix_reg)
    pp_sampler = sampler.CoauthorGraphSampler()
    pd_sampler = sampler.BipartiteGraphSampler()
    dd_sampler = sampler.LinkedDocGraphSampler()
    eval_f1 = eval_metric.Evaluator()

    run_helper = train_helper.TrainHelper()
    pairwise_precision, pairwise_recall, pairwise_f1 = run_helper.helper(num_epoch, dataset, bpr_optimizer,
                                                                         pp_sampler, pd_sampler, dd_sampler,
                                                                         eval_f1, sampler_method)

    return paper_count, pairwise_precision, pairwise_recall, pairwise_f1


if __name__ == "__main__":
    # args = parse_args()


    latent_dimen = 20
    alpha = 0.02
    matrix_reg = 0.01
    num_epoch = 100
    sampler_method = 'uniform'

    file_list = get_file_list('E:\\Works\\name_disambiguation\\dataset', [])

    total_papers_count = 0
    avg_pairwise_precision = 0.0
    avg_pairwise_recall = 0.0
    avg_pairwise_f1 = 0.0

    # file_list = ['E:\\Works\\name_disambiguation\\dataset\\F. Wang.xml']
    # file_list = ['E:\\Works\\name_disambiguation\\dataset\\Yang Wang.xml']
    for file_path in file_list:
        author_name = file_path.split('\\')[-1].replace('.xml', '')
        author_name = author_name.lower()
        author_name = re.sub('[^A-Za-z0-9]', ' ', author_name)
        author_name = re.sub('\s{2,}', ' ', author_name)

        print author_name,

        paper_count, pairwise_precision, pairwise_recall, pairwise_f1 = main(file_path, latent_dimen, alpha, matrix_reg, num_epoch,
                                                                sampler_method)

        avg_pairwise_precision += pairwise_precision
        avg_pairwise_recall += pairwise_recall
        avg_pairwise_f1 += pairwise_f1
        total_papers_count += paper_count

        print '\t %d' % paper_count,
        print '\t %f' % pairwise_precision,
        print '\t %f' % pairwise_recall,
        print '\t %f' % pairwise_f1

    print len(file_list)
    print total_papers_count
    avg_pairwise_precision /= len(file_list)
    avg_pairwise_recall /= len(file_list)
    avg_pairwise_f1 /= len(file_list)
    print 'avg_pairwise_precision: %f' % avg_pairwise_precision
    print 'avg_pairwise_recall: %f' % avg_pairwise_recall
    print 'avg_pairwise_f1: %f' % avg_pairwise_f1
