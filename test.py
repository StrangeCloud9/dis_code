
import networkx as nx
import MySQLdb
from lxml import html

if __name__ == "__main__":
    # args = parse_args()


    latent_dimen = 20
    alpha = 0.02
    matrix_reg = 0.01
    num_epoch = 100
    sampler_method = 'uniform'

    file_path = 'E:\\Works\\name_disambiguation\\dataset\\Yang Wang.xml'

    tree = html.parse(file_path)
    root = tree.getroot()

    print root.xpath('//person')[0].xpath('//personID')