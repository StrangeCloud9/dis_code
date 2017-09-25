class TrainHelper():
    @staticmethod
    def helper(num_epoch, dataset, bpr_optimizer,
               pp_sampler, pd_sampler, dd_sampler,
               eval_f1, sampler_method):

        pairwise_precision = 0.0
        pairwise_recall = 0.0
        pairwise_f1 = 0.0

        bpr_optimizer.init_model(dataset)
        if sampler_method == "uniform":
            unchanged_time = 0
            old_pairwise_f1 = 0.0
            for _ in xrange(0, num_epoch):
                bpr_loss = 0.0
                for _ in xrange(0, dataset.num_nnz):
                    """
                    update embedding in person-person network
                    update embedding in person-document network
                    update embedding in doc-doc network
                    """
                    if len(dataset.C_Graph.nodes()) != 0:
                        for i, j, t in pp_sampler.generate_triplet_uniform(dataset):
                            bpr_optimizer.update_pp_gradient(i, j, t)
                            bpr_loss += bpr_optimizer.compute_pp_loss(i, j, t)

                    if len(dataset.C_Graph.nodes()) != 0:
                        for i, j, t in pd_sampler.generate_triplet_uniform(dataset):
                            bpr_optimizer.update_pd_gradient(i, j, t)
                            bpr_loss += bpr_optimizer.compute_pd_loss(i, j, t)

                    if len(dataset.D_Graph.nodes()) != 0:
                        for i, j, t in dd_sampler.generate_triplet_uniform(dataset):
                            bpr_optimizer.update_dd_gradient(i, j, t)
                            bpr_loss += bpr_optimizer.compute_dd_loss(i, j, t)

                # average_loss = float(bpr_loss) / dataset.num_nnz
                # print "average bpr loss is " + str(average_loss)
                pairwise_precision, pairwise_recall, pairwise_f1 = eval_f1.compute_f1(dataset, bpr_optimizer)
                # print 'auc is ' + str(pairwise_f1)

                if abs(pairwise_f1 - old_pairwise_f1) < 0.00001:
                    unchanged_time += 1
                else:
                    unchanged_time = 0
                if unchanged_time > 10:
                    break
                old_pairwise_f1 = pairwise_f1

        elif sampler_method == "reject":
            for _ in xrange(0, num_epoch):
                # bpr_loss = 0.0
                for _ in xrange(0, dataset.num_nnz):
                    """
                    update embedding in person-person network
                    update embedding in person-document network
                    update embedding in doc-doc network
                    """
                    for i, j, t in pp_sampler.generate_triplet_reject(dataset, bpr_optimizer):
                        bpr_optimizer.update_pp_gradient(i, j, t)
                        # bpr_loss += bpr_optimizer.compute_pp_loss(i, j, t)

                    for i, j, t in pd_sampler.generate_triplet_reject(dataset, bpr_optimizer):
                        bpr_optimizer.update_pd_gradient(i, j, t)
                        # bpr_loss += bpr_optimizer.compute_pd_loss(i, j, t)

                    for i, j, t in dd_sampler.generate_triplet_reject(dataset, bpr_optimizer):
                        bpr_optimizer.update_dd_gradient(i, j, t)
                        # bpr_loss += bpr_optimizer.compute_dd_loss(i, j, t)

                # average_loss = float(bpr_loss) / dataset.num_nnz
                # print "average bpr loss is " + str(average_loss)
                average_f1 = eval_f1.compute_f1(dataset, bpr_optimizer)
                print 'auc is ' + str(average_f1)

        elif sampler_method == "adaptive":
            for _ in xrange(0, num_epoch):
                # bpr_loss = 0.0
                for _ in xrange(0, dataset.num_nnz):
                    """
                    update embedding in person-person network
                    update embedding in person-document network
                    update embedding in doc-doc network
                    """
                    for i, j, t in pp_sampler.generate_triplet_adaptive(dataset, bpr_optimizer):
                        bpr_optimizer.update_pp_gradient(i, j, t)
                        # bpr_loss += bpr_optimizer.compute_pp_loss(i, j, t)

                    for i, j, t in pd_sampler.generate_triplet_adaptive(dataset, bpr_optimizer):
                        bpr_optimizer.update_pd_gradient(i, j, t)
                        # bpr_loss += bpr_optimizer.compute_pd_loss(i, j, t)

                    for i, j, t in dd_sampler.generate_triplet_adaptive(dataset, bpr_optimizer):
                        bpr_optimizer.update_dd_gradient(i, j, t)
                        # bpr_loss += bpr_optimizer.compute_dd_loss(i, j, t)

                # average_loss = float(bpr_loss) / dataset.num_nnz
                # print "average bpr loss is " + str(average_loss)
                average_f1 = eval_f1.compute_f1(dataset, bpr_optimizer)
                print 'auc is ' + str(average_f1)

                #        average_f1 = eval_f1.compute_f1(dataset, bpr_optimizer)
                #        print str(average_f1)

        return pairwise_precision, pairwise_recall, pairwise_f1
