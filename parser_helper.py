import networkx as nx
import MySQLdb
from lxml import html
import re


class DataSet():

    def __init__(self, file_path):
        self.file_path = file_path
        self.paper_authorlist_dict = {}
        self.paper_list = []
        self.coauthor_list = []
        self.label_list = []
        self.C_Graph = nx.Graph()
        self.D_Graph = nx.Graph()
        self.num_nnz = 0

    def reader(self):
        conn = MySQLdb.connect(host='202.120.36.29', port=6033, user='groupleader', passwd='onlyleaders',
                               db='mag-new-160205',
                               charset="utf8")
        cursor = conn.cursor()

        author_name = self.file_path.split('\\')[-1].replace('.xml', '')
        author_name = author_name.lower()
        author_name = re.sub('[^A-Za-z0-9]', ' ', author_name)
        author_name = re.sub('\s{2,}', ' ', author_name)

        paper_index = 0
        coauthor_set = set()

        tree = html.parse(self.file_path)
        root = tree.getroot()
        for node in root.xpath('//publication'):
            label = node.xpath('label')[0].text
            title = node.xpath('title')[0].text

            title = title.lower()
            if title[-1] == '.':
                title = title[:-1]
            title = re.sub('[^A-Za-z0-9]', ' ', title)
            title = re.sub('\s{2,}', ' ', title)
            quest_paper_by_title = 'SELECT PaperID FROM Papers WHERE NormalizedPaperTitle="%s"'
            cursor.execute(quest_paper_by_title % title)
            ps = cursor.fetchall()

            paper_ids = list()
            if len(ps) == 1:
                paper_ids.append(ps[0][0])
            if len(ps) >= 2:
                for p in ps:
                    quest_author_by_paper = 'SELECT AuthorName FROM Authors INNER JOIN' \
                                            '	(SELECT AuthorID FROM PaperAuthorAffiliations AS PAA  WHERE PaperID="%s") AS TB2' \
                                            '	ON Authors.AuthorID = TB2.AuthorID'
                    cursor.execute(quest_author_by_paper % p[0])
                    authors = cursor.fetchall()
                    for author in authors:
                        if author[0] == author_name.lower():
                            paper_ids.append(p[0])

            for paper_id in paper_ids:
                paper_index += 1
                self.paper_list.append(paper_index)
                self.label_list.append(label)

                # get affiliation and coauthors
                quest_affiliation = 'SELECT AuthorName,AffiliationID FROM Authors INNER JOIN' \
                                    '	(SELECT AuthorID,AffiliationID FROM PaperAuthorAffiliations WHERE PaperID="%s") AS TB ' \
                                    'ON Authors.AuthorID = TB.AuthorID'
                cursor.execute(quest_affiliation % paper_id)
                author_affiliations = cursor.fetchall()

                himself = None
                for ai in range(len(author_affiliations)):
                    if author_affiliations[ai][0] == author_name.lower():
                        himself = ai
                        break

                if himself is None:
                    tmp1 = author_name.split()
                    count = 0
                    for ai in range(len(author_affiliations)):
                        tmp2 = author_affiliations[ai][0].split()
                        if tmp1[-1] == tmp2[-1] and tmp1[0][0] == tmp2[0][0]:
                            himself = ai
                            break
                        elif tmp1[-1] == tmp2[0] and tmp1[0][0] == tmp2[-1][0]:
                            himself = ai
                            break

                # get coauthors
                coauthor_list = list()
                for ai in range(len(author_affiliations)):
                    if ai != himself:
                        coauthor_name = author_affiliations[ai][0]
                        coauthor_list.append(coauthor_name)

                self.paper_authorlist_dict[paper_index] = coauthor_list

                for co_author in coauthor_list:
                    coauthor_set.add(co_author)

                # construct the coauthorship graph
                for pos in xrange(0, len(coauthor_list) - 1):
                    for inpos in xrange(pos + 1, len(coauthor_list)):
                        src_node = coauthor_list[pos]
                        dest_node = coauthor_list[inpos]
                        if not self.C_Graph.has_edge(src_node, dest_node):
                            self.C_Graph.add_edge(src_node, dest_node, weight=1)
                        else:
                            edge_weight = self.C_Graph[src_node][dest_node]['weight']
                            edge_weight += 1
                            self.C_Graph[src_node][dest_node]['weight'] = edge_weight

        # with open(self.file_path, "r") as filetoread:
        #     for line in filetoread:
        #         line = line.strip()
        #         if "FullName" in line:
        #             ego_name = line[line.find('>')+1:line.rfind('<')].strip()
        #         elif "<publication>" in line:
        #             paper_index += 1
        #             self.paper_list.append(paper_index)
        #         elif "<authors>" in line:
        #             author_list = line[line.find('>')+1: line.rfind('<')].strip().split(',')
        #             if len(author_list) > 1:
        #                 if ego_name in author_list:
        #                     author_list.remove(ego_name)
        #                     self.paper_authorlist_dict[paper_index] = author_list
        #                 else:
        #                     self.paper_authorlist_dict[paper_index] = author_list
        #
        #                 for co_author in author_list:
        #                     coauthor_set.add(co_author)
        #
        #                 # construct the coauthorship graph
        #                 for pos in xrange(0, len(author_list) - 1):
        #                     for inpos in xrange(pos+1, len(author_list)):
        #                         src_node = author_list[pos]
        #                         dest_node = author_list[inpos]
        #                         if not self.C_Graph.has_edge(src_node, dest_node):
        #                             self.C_Graph.add_edge(src_node, dest_node, weight = 1)
        #                         else:
        #                             edge_weight = self.C_Graph[src_node][dest_node]['weight']
        #                             edge_weight += 1
        #                             self.C_Graph[src_node][dest_node]['weight'] = edge_weight
        #             else:
        #                 self.paper_authorlist_dict[paper_index] = []
        #         elif "<label>" in line:
        #             label = int(line[line.find('>')+1: line.rfind('<')].strip())
        #             self.label_list.append(label)

        self.coauthor_list = list(coauthor_set)
        """
        compute the 2-extension coauthorship for each paper
        generate doc-doc network
        edge weight is based on 2-coauthorship relation
        edge weight details are in paper definition 3.3
        """
        paper_2hop_dict = {}
        for paper_idx in self.paper_list:
            temp = set()
            if self.paper_authorlist_dict[paper_idx] != []:
                for first_hop in self.paper_authorlist_dict[paper_idx]:
                    temp.add(first_hop)
                    if self.C_Graph.has_node(first_hop):
                        for snd_hop in self.C_Graph.neighbors(first_hop):
                            temp.add(snd_hop)
            paper_2hop_dict[paper_idx] = temp

        for idx1 in xrange(0, len(self.paper_list) - 1):
            for idx2 in xrange(idx1 + 1, len(self.paper_list)):
                temp_set1 = paper_2hop_dict[self.paper_list[idx1]]
                temp_set2 = paper_2hop_dict[self.paper_list[idx2]]

                edge_weight = len(temp_set1.intersection(temp_set2))
                if edge_weight != 0:
                    self.D_Graph.add_edge(self.paper_list[idx1],
                                          self.paper_list[idx2],
                                          weight = edge_weight)
        bipartite_num_edge = 0
        for key, val in self.paper_authorlist_dict.items():
            if val != []:
                bipartite_num_edge += len(val)

        self.num_nnz = self.D_Graph.number_of_edges() + \
                       self.C_Graph.number_of_edges() + \
                       bipartite_num_edge

        cursor.close()
        conn.close()

        # print len(self.D_Graph.nodes())
        return paper_index
